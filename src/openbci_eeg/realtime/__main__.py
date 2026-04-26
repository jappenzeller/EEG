"""Entry point: wire the board, ring buffer, acquisition thread, HDF5 writer,
and the Qt main window together.

Usage:
    python -m openbci_eeg.realtime --mode cyton_daisy --serial-port COM4
    python -m openbci_eeg.realtime --sq-profile scalp --output session.h5
    python -m openbci_eeg.realtime --sq-log --session-notes "O1 O2 test"

Threading layout:
    Qt main thread    : UI (QTimer @ 30 Hz pulls from ring)
    Acquisition thread: Board.poll() -> RingBuffer.push() -> writer callback
    Writer thread     : drain queue -> HDF5 flush
    (optional) SQ log : 1 Hz print of per-channel signal quality to stdout
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
import threading
from pathlib import Path

from PySide6 import QtWidgets

from openbci_eeg.acquisition.board import Board, RealtimeBoardConfig, BoardMode
from .ring_buffer import RingBuffer
from .acquisition_thread import AcquisitionThread
from .io.hdf5_writer import HDF5Writer
from .ui.main_window import MainWindow
from .analysis.signal_quality import PROFILES, BENCH, CHANNEL_POSITIONS

MONTAGE_VERSION = "2026-04-26-o2-on-fast-channel"

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="openbci-eeg realtime")
    p.add_argument(
        "--mode",
        choices=[m.value for m in BoardMode],
        default=BoardMode.CYTON_DAISY.value,
    )
    p.add_argument("--serial-port", default="COM4", help="e.g. /dev/ttyUSB0 or COM4")
    p.add_argument("--playback-file", default=None, help="HDF5 file for PLAYBACK mode")
    p.add_argument("--master-board", type=int, default=2,
                    help="original board_id of playback file (2 = CYTON_DAISY)")
    p.add_argument("--output", default=None, help="HDF5 output session path (optional)")
    p.add_argument("--buffer-sec", type=float, default=30.0, help="ring buffer length")
    p.add_argument("--poll-ms", type=int, default=50, help="acquisition poll interval")
    p.add_argument(
        "--sq-profile",
        choices=list(PROFILES.keys()),
        default="bench",
        help="Signal quality threshold profile (bench=SRB-only, scalp=BIAS+paste)",
    )
    p.add_argument("--sq-log", action="store_true",
                    help="Print per-channel signal quality to stdout at 1 Hz")
    p.add_argument("--session-notes", default="",
                    help="Free-text notes written to HDF5 root attrs")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _sq_log_thread(
    ring: RingBuffer,
    sample_rate: float,
    ch_names: list[str],
    profile_name: str,
    stop_event: threading.Event,
) -> None:
    """Print per-channel signal quality at 1 Hz."""
    from .analysis.signal_quality import railed_percent, channel_std, line_noise_ratio, PROFILES

    profile = PROFILES[profile_name]
    window_samples = int(sample_rate * 2.0)
    t0 = time.monotonic()

    while not stop_event.is_set():
        stop_event.wait(1.0)
        if stop_event.is_set():
            break

        data, _ = ring.get_latest(window_samples)
        if data.shape[1] == 0:
            continue

        pcts = railed_percent(data, profile.rail_threshold_uv)
        stds = channel_std(data)
        lines = line_noise_ratio(data, sample_rate)
        elapsed = time.monotonic() - t0

        parts = [f"[sq] t={elapsed:6.1f}"]
        for i, name in enumerate(ch_names):
            if i < len(pcts):
                parts.append(
                    f"  {name:4s} r={pcts[i]:4.0f} s={stds[i]:6.1f} l={lines[i]*100:3.0f}%"
                )
        print("".join(parts), flush=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    profile = PROFILES[args.sq_profile]
    print(f"[sq] Profile: {profile.name} -- {profile.description}", flush=True)

    # For synthetic mode, don't require a serial port
    serial_port = None if args.mode == BoardMode.SYNTHETIC.value else args.serial_port

    config = RealtimeBoardConfig(
        mode=BoardMode(args.mode),
        serial_port=serial_port,
        playback_file=args.playback_file,
        master_board=args.master_board if args.mode == BoardMode.PLAYBACK.value else None,
    )

    board = Board(config)
    board.prepare()

    sr = board.sample_rate
    n_ch = board.n_eeg_channels
    # Use canonical montage positions, not BrainFlow defaults
    ch_names = [CHANNEL_POSITIONS[i] for i in range(1, n_ch + 1)]
    log.info("board ready: mode=%s, sr=%d, channels=%d", args.mode, sr, n_ch)
    log.info("using custom montage: %s", ch_names)

    ring = RingBuffer(n_ch, int(args.buffer_sec * sr))

    writer: HDF5Writer | None = None
    if args.output:
        writer = HDF5Writer(
            Path(args.output),
            n_channels=n_ch,
            sample_rate=sr,
            channel_names=ch_names,
            metadata={
                "mode": args.mode,
                "sq_profile": args.sq_profile,
                "session_notes": args.session_notes,
                "montage_version": MONTAGE_VERSION,
            },
        )
        writer.start()

    acq = AcquisitionThread(board, ring, poll_interval_sec=args.poll_ms / 1000.0)
    if writer is not None:
        acq.add_callback(writer.push_eeg)

    board.start()
    acq.start()

    # Marker function for paradigm runner
    def mark_fn(marker_id: int, label: str) -> None:
        board.insert_marker(float(marker_id))
        if writer is not None:
            writer.push_marker(ring.total_samples, marker_id, label)
        log.info("marker: %d %s (sample=%d)", marker_id, label, ring.total_samples)

    # Optional signal quality logger
    sq_stop = threading.Event()
    sq_thread: threading.Thread | None = None
    if args.sq_log:
        sq_thread = threading.Thread(
            target=_sq_log_thread,
            args=(ring, sr, ch_names, args.sq_profile, sq_stop),
            daemon=True,
            name="SQLog",
        )
        sq_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("openbci-lattice")
    win = MainWindow(ring, sr, ch_names, profile=profile, mark_fn=mark_fn)
    win.show()

    signal.signal(signal.SIGINT, lambda *_: app.quit())

    rc = 0
    try:
        rc = app.exec()
    finally:
        log.info("shutting down...")
        sq_stop.set()
        if sq_thread is not None:
            sq_thread.join(timeout=2.0)
        acq.stop()
        acq.join(timeout=2.0)
        board.stop()
        board.release()
        if writer is not None:
            writer.stop()
            writer.join(timeout=10.0)
    sys.exit(rc)


if __name__ == "__main__":
    main()
