"""
CLI interface for OpenBCI EEG pipeline.

Usage:
    openbci-eeg record --duration 60 --subject S001
    openbci-eeg test-board
    openbci-eeg preprocess data/raw/session_001/
    openbci-eeg extract-pn data/processed/session_001/
    openbci-eeg upload --subject S001 --session 20260215
    openbci-eeg pipeline --config configs/default.yaml
"""

from __future__ import annotations

import logging
import sys

import click

from openbci_eeg import __version__


@click.group()
@click.version_option(version=__version__)
@click.option("--config", "-c", default=None, help="Path to YAML config file.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """OpenBCI EEG acquisition and processing pipeline for QDNU."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)

    from openbci_eeg.config import load_config
    ctx.obj["config"] = load_config(config)


@main.command()
@click.option("--duration", "-d", default=60.0, help="Recording duration in seconds.")
@click.option("--subject", "-s", default="S000", help="Subject ID.")
@click.option("--session-type", "-t", default="resting", help="Session type.")
@click.option("--port", "-p", default="COM4", help="Serial port (default: COM4).")
@click.option("--synthetic", is_flag=True, help="Use synthetic board (no hardware).")
@click.pass_context
def record(ctx, duration, subject, session_type, port, synthetic):
    """Record an EEG session."""
    from openbci_eeg.acquisition.board import Board, RealtimeBoardConfig, BoardMode, record as do_record

    config = ctx.obj["config"]
    config.session.subject_id = subject
    config.session.session_type = session_type
    config.session.duration_sec = duration

    mode = BoardMode.SYNTHETIC if synthetic else BoardMode.CYTON_DAISY
    board_config = RealtimeBoardConfig(
        mode=mode,
        serial_port=None if synthetic else port,
    )

    click.echo(f"Connecting to {'synthetic' if synthetic else f'Cyton+Daisy on {port}'}...")
    board = Board(board_config)
    try:
        board.prepare()
        click.echo(f"Connected: {board.n_eeg_channels} channels @ {board.sample_rate} Hz")
        output_dir = config.session.data_dir / "raw"
        data = do_record(board.shim, duration, output_dir=output_dir)
        click.echo(f"Recorded {data.shape[1]} samples to {output_dir}")
    finally:
        board.release()


@main.command("test-board")
@click.option("--port", "-p", default="COM4", help="Serial port (default: COM4).")
@click.option("--synthetic", is_flag=True, help="Use synthetic board.")
@click.option("--duration", "-d", default=10.0, help="Test duration in seconds.")
@click.pass_context
def test_board(ctx, port, synthetic, duration):
    """Verify board connection and stream test data."""
    import time
    import numpy as np
    from openbci_eeg.acquisition.board import Board, RealtimeBoardConfig, BoardMode

    mode = BoardMode.SYNTHETIC if synthetic else BoardMode.CYTON_DAISY
    config = RealtimeBoardConfig(
        mode=mode,
        serial_port=None if synthetic else port,
    )

    click.echo(f"Connecting to {'synthetic' if synthetic else f'Cyton+Daisy on {port}'}...")
    board = Board(config)

    try:
        board.prepare()
        click.echo(f"Connected: {board.n_eeg_channels} channels @ {board.sample_rate} Hz")

        click.echo(f"Streaming for {duration:.0f} seconds...")
        board.start()
        time.sleep(duration)
        eeg, ts = board.poll()
        board.stop()

        n_samples = eeg.shape[1]
        actual_sec = n_samples / board.sample_rate
        click.echo(f"Received: {n_samples} samples ({actual_sec:.1f} s)")
        click.echo(f"Data range: {eeg.min():.1f} to {eeg.max():.1f} uV")
        click.echo("")

        # Per-channel report
        ch_names = board.channel_names
        for i in range(eeg.shape[0]):
            ch = eeg[i]
            name = ch_names[i] if i < len(ch_names) else f"CH{i+1}"
            std = ch.std()
            status = "OK" if 1.0 < std < 500.0 else "FLAT" if std < 1.0 else "NOISY"
            click.echo(f"  {name:4s}: mean={ch.mean():8.1f}  std={std:8.1f}  [{status}]")

        # Summary
        stds = np.array([eeg[i].std() for i in range(eeg.shape[0])])
        good = np.sum((stds > 1.0) & (stds < 500.0))
        click.echo(f"\n{good}/{eeg.shape[0]} channels OK")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    finally:
        board.release()
        click.echo("Board released.")


@main.command()
@click.argument("input_dir")
@click.option("--output", "-o", default=None, help="Output directory.")
@click.pass_context
def preprocess(ctx, input_dir, output):
    """Preprocess a raw recording."""
    from pathlib import Path
    from openbci_eeg.preprocessing.convert import load_recording_to_mne
    from openbci_eeg.preprocessing.filters import preprocess_raw

    config = ctx.obj["config"]
    raw = load_recording_to_mne(input_dir)
    raw_clean = preprocess_raw(raw, config.preprocess)

    if output:
        out_path = Path(output)
    else:
        out_path = Path(input_dir).parent.parent / "processed" / Path(input_dir).name

    out_path.mkdir(parents=True, exist_ok=True)
    raw_clean.save(out_path / "preprocessed-raw.fif", overwrite=True)
    click.echo(f"Preprocessed data saved to {out_path}")


@main.command("extract-pn")
@click.argument("input_dir")
@click.option("--output", "-o", default=None, help="Output path for .npz file.")
@click.pass_context
def extract_pn(ctx, input_dir, output):
    """Extract PN parameters from preprocessed data."""
    from pathlib import Path
    import mne
    from openbci_eeg.preprocessing.convert import mne_to_pn_input
    from openbci_eeg.pn_extraction import extract_pn_multichannel, save_pn_parameters

    config = ctx.obj["config"]
    input_dir = Path(input_dir)

    # Load preprocessed data
    fif_path = input_dir / "preprocessed-raw.fif"
    if fif_path.exists():
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    else:
        from openbci_eeg.preprocessing.convert import load_recording_to_mne
        raw = load_recording_to_mne(input_dir)

    eeg_uv, sfreq, ch_names = mne_to_pn_input(raw)
    pn_params = extract_pn_multichannel(eeg_uv, sfreq, ch_names, config.pn)

    if output:
        out_path = Path(output)
    else:
        out_path = input_dir / "pn_parameters.npz"

    save_pn_parameters(pn_params, out_path)
    click.echo(f"PN parameters saved to {out_path}")


@main.command()
@click.argument("local_dir")
@click.option("--subject", "-s", required=True, help="Subject ID.")
@click.option("--session", required=True, help="Session ID.")
@click.pass_context
def upload(ctx, local_dir, subject, session):
    """Upload a session to S3."""
    from openbci_eeg.aws.storage import upload_session
    config = ctx.obj["config"]
    uri = upload_session(local_dir, subject, session, config=config.aws)
    click.echo(f"Uploaded to {uri}")


@main.command()
@click.option("--duration", "-d", default=60.0, help="Recording duration.")
@click.option("--subject", "-s", default="S000", help="Subject ID.")
@click.option("--port", "-p", default=None, help="Serial port.")
@click.option("--synthetic", is_flag=True, help="Use synthetic board.")
@click.option("--skip-upload", is_flag=True, help="Skip S3 upload.")
@click.pass_context
def pipeline(ctx, duration, subject, port, synthetic, skip_upload):
    """Run full pipeline: record -> preprocess -> extract PN -> upload."""
    click.echo("Pipeline: record -> preprocess -> extract-pn -> upload")
    click.echo("(Not yet implemented -- run individual commands for now)")
    # TODO: chain record, preprocess, extract_pn, upload


@main.command()
@click.option("--mode", "-m", default="synthetic",
              type=click.Choice(["synthetic", "cyton_daisy", "playback"]),
              help="Board mode.")
@click.option("--port", "-p", default=None, help="Serial port for cyton_daisy mode.")
@click.option("--output", "-o", default=None, help="HDF5 output file path.")
@click.option("--playback-file", default=None, help="HDF5 file for playback mode.")
@click.option("--buffer-sec", default=30.0, help="Ring buffer length in seconds.")
def lattice(mode, port, output, playback_file, buffer_sec):
    """Launch real-time EEG display (Qt UI with scrolling time series)."""
    import sys
    sys.argv = ["lattice"]  # reset argv before Qt parses it
    if mode:
        sys.argv.extend(["--mode", mode])
    if port:
        sys.argv.extend(["--serial-port", port])
    if output:
        sys.argv.extend(["--output", output])
    if playback_file:
        sys.argv.extend(["--playback-file", playback_file])
    sys.argv.extend(["--buffer-sec", str(buffer_sec)])

    from openbci_eeg.realtime.__main__ import main as realtime_main
    realtime_main()


if __name__ == "__main__":
    main()
