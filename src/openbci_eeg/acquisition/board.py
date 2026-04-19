"""
OpenBCI Cyton+Daisy board connection and data streaming via BrainFlow.

Provides both a class-based API (`Board`) for use by the realtime subpackage
and a functional API (`connect`, `disconnect`, `record`, `stream`) for the
CLI and batch pipeline.

Class-based (realtime / advanced):
    board = Board(BoardConfig(mode=BoardMode.SYNTHETIC))
    board.prepare()
    board.start()
    eeg, ts = board.poll()
    board.stop()
    board.release()

Functional (CLI / batch):
    board = connect(config, synthetic=True)
    data = record(board, duration_sec=60)
    disconnect(board)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from openbci_eeg.config import BoardConfig as PipelineBoardConfig, PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class-based API (shared core)
# ---------------------------------------------------------------------------

class BoardMode(str, Enum):
    SYNTHETIC = "synthetic"
    CYTON_DAISY = "cyton_daisy"
    PLAYBACK = "playback"


@dataclass
class RealtimeBoardConfig:
    """Board configuration for the realtime layer."""
    mode: BoardMode = BoardMode.SYNTHETIC
    serial_port: Optional[str] = None
    playback_file: Optional[str] = None
    master_board: Optional[int] = None


class Board:
    """Typed wrapper around BrainFlow's BoardShim.

    Used directly by the realtime subpackage (acquisition thread, HDF5 writer,
    UI). The functional API below delegates to this class.
    """

    def __init__(self, config: RealtimeBoardConfig) -> None:
        self._config = config
        params = BrainFlowInputParams()

        if config.mode == BoardMode.SYNTHETIC:
            self.board_id = BoardIds.SYNTHETIC_BOARD.value
        elif config.mode == BoardMode.CYTON_DAISY:
            if not config.serial_port:
                raise ValueError("serial_port is required for CYTON_DAISY mode")
            self.board_id = BoardIds.CYTON_DAISY_BOARD.value
            params.serial_port = config.serial_port
        elif config.mode == BoardMode.PLAYBACK:
            if not config.playback_file or config.master_board is None:
                raise ValueError("playback_file and master_board required for PLAYBACK mode")
            self.board_id = BoardIds.PLAYBACK_FILE_BOARD.value
            params.file = config.playback_file
            params.master_board = config.master_board
        else:
            raise ValueError(f"unknown board mode: {config.mode}")

        # Cyton+Daisy can take 10-15s to handshake over RF
        if config.mode == BoardMode.CYTON_DAISY:
            params.timeout = 30

        self.params = params
        self._shim: Optional[BoardShim] = None

    # --- lifecycle ---------------------------------------------------------

    def prepare(self) -> None:
        self._shim = BoardShim(self.board_id, self.params)
        try:
            self._shim.prepare_session()
        except Exception:
            # Release immediately so the COM port isn't left locked
            try:
                self._shim.release_session()
            except Exception:
                pass
            self._shim = None
            raise

    def start(self, buffer_samples: int = 450_000) -> None:
        assert self._shim is not None, "call prepare() first"
        self._shim.start_stream(buffer_samples)

    def stop(self) -> None:
        if self._shim is not None and self._shim.is_prepared():
            try:
                self._shim.stop_stream()
            except Exception:
                pass

    def release(self) -> None:
        if self._shim is not None:
            try:
                self._shim.release_session()
            finally:
                self._shim = None

    # --- streaming ---------------------------------------------------------

    def poll(self) -> tuple[np.ndarray, np.ndarray]:
        """Pull all available samples since last call.

        Returns (eeg_uV, timestamps). eeg shape = (n_eeg_channels, n_samples).
        """
        assert self._shim is not None
        raw = self._shim.get_board_data()
        if raw.shape[1] == 0:
            return (
                np.zeros((self.n_eeg_channels, 0), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
            )
        eeg = raw[self.eeg_channel_indices, :].astype(np.float32, copy=False)
        ts = raw[self.timestamp_channel, :].astype(np.float64, copy=False)
        return eeg, ts

    def get_all_data(self) -> np.ndarray:
        """Get all data from board buffer (clears buffer)."""
        assert self._shim is not None
        return self._shim.get_board_data()

    def insert_marker(self, value: float) -> None:
        """Insert a marker into the stream."""
        assert self._shim is not None
        self._shim.insert_marker(float(value))

    def config_board(self, cmd: str) -> str:
        """Send a raw config string to the board."""
        assert self._shim is not None
        return self._shim.config_board(cmd)

    # --- metadata ---------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return BoardShim.get_sampling_rate(self.board_id)

    @property
    def eeg_channel_indices(self) -> list[int]:
        return BoardShim.get_eeg_channels(self.board_id)

    @property
    def n_eeg_channels(self) -> int:
        return len(self.eeg_channel_indices)

    @property
    def timestamp_channel(self) -> int:
        return BoardShim.get_timestamp_channel(self.board_id)

    @property
    def marker_channel(self) -> int:
        return BoardShim.get_marker_channel(self.board_id)

    @property
    def channel_names(self) -> list[str]:
        try:
            return BoardShim.get_eeg_names(self.board_id)
        except Exception:
            return [f"CH{i + 1}" for i in range(self.n_eeg_channels)]

    @property
    def shim(self) -> BoardShim:
        """Access the underlying BoardShim for advanced operations."""
        assert self._shim is not None
        return self._shim


# ---------------------------------------------------------------------------
# Functional API (backwards-compatible, used by CLI)
# ---------------------------------------------------------------------------

def connect(
    config: Optional[PipelineBoardConfig | PipelineConfig] = None,
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
        config = PipelineBoardConfig()
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
            "synthetic board" if synthetic else "Cyton+Daisy",
            params.serial_port or "N/A",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to board: {e}") from e

    return board


def disconnect(board: BoardShim) -> None:
    """Safely disconnect from board. Stops stream if active."""
    try:
        if board.is_prepared():
            board.stop_stream()
    except Exception:
        pass

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
    samples have accumulated.
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
    """Attempt to auto-detect the OpenBCI dongle serial port."""
    import glob
    import platform

    system = platform.system()

    if system == "Linux":
        candidates = glob.glob("/dev/ttyUSB*")
    elif system == "Darwin":
        candidates = glob.glob("/dev/tty.usbserial-*")
    elif system == "Windows":
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

    np.save(output_dir / "raw_data.npy", data)

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
