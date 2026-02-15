"""
OpenBCI Cyton+Daisy board connection and data streaming via BrainFlow.

Usage:
    board = connect(config)
    data = record(board, duration_sec=60)
    disconnect(board)

Or for continuous streaming:
    board = connect(config)
    for chunk in stream(board, chunk_samples=256):
        process(chunk)
    disconnect(board)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from openbci_eeg.config import BoardConfig, PipelineConfig

logger = logging.getLogger(__name__)


def connect(
    config: Optional[BoardConfig | PipelineConfig] = None,
    serial_port: Optional[str] = None,
    synthetic: bool = False,
) -> BoardShim:
    """
    Connect to OpenBCI Cyton+Daisy board.

    Args:
        config: Board configuration. Uses defaults if None.
        serial_port: Override serial port (takes precedence over config).
        synthetic: If True, use synthetic board for testing (no hardware needed).

    Returns:
        Connected BoardShim instance ready for streaming.

    Raises:
        RuntimeError: If connection fails after retries.
    """
    if config is None:
        config = BoardConfig()
    elif isinstance(config, PipelineConfig):
        config = config.board

    if synthetic:
        board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        board_id = config.board_id

    params = BrainFlowInputParams()

    if not synthetic:
        port = serial_port or config.serial_port
        if port is None:
            port = _auto_detect_port()
        params.serial_port = port

    if config.log_level > 0:
        BoardShim.enable_dev_board_logger()

    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
        logger.info(
            "Connected to %s on %s",
            "synthetic board" if synthetic else f"Cyton+Daisy",
            params.serial_port or "N/A",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to board: {e}") from e

    return board


def disconnect(board: BoardShim) -> None:
    """
    Safely disconnect from board. Stops stream if active.
    """
    try:
        if board.is_prepared():
            board.stop_stream()
    except Exception:
        pass  # Stream may not have been started

    try:
        board.release_session()
        logger.info("Board disconnected.")
    except Exception as e:
        logger.warning("Error during disconnect: %s", e)


def record(
    board: BoardShim,
    duration_sec: float,
    output_dir: Optional[str | Path] = None,
    session_id: Optional[str] = None,
) -> np.ndarray:
    """
    Record a fixed-duration session.

    Args:
        board: Connected BoardShim instance.
        duration_sec: Recording duration in seconds.
        output_dir: If provided, saves .npy and metadata .json to this directory.
        session_id: Optional session identifier for metadata.

    Returns:
        Raw data array of shape (channels, samples).
    """
    board_id = board.get_board_id()
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sample_rate = BoardShim.get_sampling_rate(board_id)

    if session_id is None:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info("Recording %0.1f seconds (session: %s)...", duration_sec, session_id)

    board.start_stream()
    time.sleep(duration_sec)
    data = board.get_board_data()
    board.stop_stream()

    logger.info(
        "Recorded %d samples across %d channels (%.1f sec actual)",
        data.shape[1],
        len(eeg_channels),
        data.shape[1] / sample_rate,
    )

    if output_dir is not None:
        _save_recording(data, board_id, session_id, output_dir)

    return data


def stream(
    board: BoardShim,
    chunk_samples: int = 256,
    poll_interval_sec: float = 0.05,
) -> Generator[np.ndarray, None, None]:
    """
    Yield data chunks continuously from the board.

    Each chunk is shape (channels, chunk_samples). Yields when enough
    samples have accumulated. Use in a for loop:

        for chunk in stream(board, chunk_samples=256):
            process(chunk)

    Args:
        board: Connected BoardShim instance.
        chunk_samples: Number of samples per yielded chunk.
        poll_interval_sec: Sleep between buffer polls.

    Yields:
        np.ndarray of shape (total_channels, chunk_samples).
    """
    board.start_stream()
    logger.info("Streaming started (chunk_size=%d).", chunk_samples)

    try:
        while True:
            count = board.get_board_data_count()
            if count >= chunk_samples:
                chunk = board.get_board_data(chunk_samples)
                yield chunk
            else:
                time.sleep(poll_interval_sec)
    except GeneratorExit:
        pass
    finally:
        board.stop_stream()
        logger.info("Streaming stopped.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_eeg_data(data: np.ndarray, board_id: int = 2) -> np.ndarray:
    """Extract only EEG channel rows from full board data array."""
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    return data[eeg_channels, :]


def get_timestamps(data: np.ndarray, board_id: int = 2) -> np.ndarray:
    """Extract timestamp row from full board data array."""
    ts_channel = BoardShim.get_timestamp_channel(board_id)
    return data[ts_channel, :]


def _auto_detect_port() -> str:
    """
    Attempt to auto-detect the OpenBCI dongle serial port.

    Returns first match from common port patterns.
    Raises RuntimeError if no port found.
    """
    import glob
    import platform

    system = platform.system()

    if system == "Linux":
        candidates = glob.glob("/dev/ttyUSB*")
    elif system == "Darwin":
        candidates = glob.glob("/dev/tty.usbserial-*")
    elif system == "Windows":
        # BrainFlow handles COM port detection on Windows;
        # fallback to common defaults
        candidates = ["COM3", "COM4", "COM5"]
    else:
        candidates = []

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect serial port. "
            "Specify it in config or pass serial_port= argument."
        )

    port = sorted(candidates)[0]
    logger.info("Auto-detected serial port: %s", port)
    return port


def _save_recording(
    data: np.ndarray,
    board_id: int,
    session_id: str,
    output_dir: str | Path,
) -> None:
    """Save raw recording and metadata to disk."""
    output_dir = Path(output_dir) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    np.save(output_dir / "raw_data.npy", data)

    # Save metadata
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    metadata = {
        "session_id": session_id,
        "board_id": board_id,
        "sample_rate": BoardShim.get_sampling_rate(board_id),
        "eeg_channel_indices": eeg_channels,
        "channel_names": BoardShim.get_eeg_names(board_id),
        "total_channels": data.shape[0],
        "total_samples": data.shape[1],
        "duration_sec": data.shape[1] / BoardShim.get_sampling_rate(board_id),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Recording saved to %s", output_dir)
