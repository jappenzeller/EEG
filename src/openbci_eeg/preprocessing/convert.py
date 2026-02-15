"""
Convert between BrainFlow data arrays and MNE-Python Raw objects.

BrainFlow outputs µV; MNE expects V. This module handles the conversion
and sets up proper channel info, montage, and metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
from brainflow.board_shim import BoardShim, BoardIds

from openbci_eeg import CHANNEL_NAMES

logger = logging.getLogger(__name__)

# BrainFlow outputs µV, MNE expects V
UV_TO_V = 1e-6


def brainflow_to_mne(
    data: np.ndarray,
    board_id: int = BoardIds.CYTON_DAISY_BOARD.value,
    channel_names: Optional[list[str]] = None,
    montage_name: str = "standard_1020",
) -> mne.io.RawArray:
    """
    Convert BrainFlow data array to MNE Raw object.

    Args:
        data: Full BrainFlow data array (all channels including non-EEG).
        board_id: BrainFlow board ID (default: Cyton+Daisy = 2).
        channel_names: Custom channel names. Uses standard 10-20 if None.
        montage_name: MNE montage name for electrode positions.

    Returns:
        MNE RawArray with proper channel info, units, and montage.
    """
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sfreq = BoardShim.get_sampling_rate(board_id)

    if channel_names is None:
        channel_names = list(CHANNEL_NAMES[:len(eeg_channels)])

    # Extract EEG data and convert µV → V
    eeg_data = data[eeg_channels, :] * UV_TO_V

    # Create MNE info
    ch_types = ["eeg"] * len(channel_names)
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=ch_types)

    # Create Raw object
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    # Set montage (electrode positions for topomaps)
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing="warn")
    except Exception as e:
        logger.warning("Could not set montage: %s", e)

    return raw


def load_recording_to_mne(
    recording_dir: str | Path,
    montage_name: str = "standard_1020",
) -> mne.io.RawArray:
    """
    Load a saved recording directory (from board.record) into MNE.

    Expects:
        recording_dir/
            raw_data.npy
            metadata.json

    Args:
        recording_dir: Path to recording directory.
        montage_name: MNE montage name.

    Returns:
        MNE RawArray.
    """
    recording_dir = Path(recording_dir)

    data = np.load(recording_dir / "raw_data.npy")
    with open(recording_dir / "metadata.json") as f:
        metadata = json.load(f)

    board_id = metadata["board_id"]
    channel_names = metadata.get("channel_names")

    return brainflow_to_mne(data, board_id, channel_names, montage_name)


def mne_to_pn_input(raw: mne.io.RawArray) -> tuple[np.ndarray, float, list[str]]:
    """
    Extract the arrays needed for PN parameter extraction from an MNE Raw.

    Returns:
        Tuple of (eeg_data_uv, sfreq, channel_names) where eeg_data_uv
        is in µV (converted back from MNE's V representation).
    """
    data_v = raw.get_data()          # V
    data_uv = data_v / UV_TO_V       # Back to µV for PN extraction
    return data_uv, raw.info["sfreq"], list(raw.ch_names)
