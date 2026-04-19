"""HDF5 <-> MNE conversion bridge.

Bridges the realtime subpackage's HDF5 session format with MNE-Python Raw
objects used by the preprocessing pipeline. This ensures recordings made
with the realtime UI feed directly into openbci_eeg.preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import mne
import numpy as np

from openbci_eeg import CHANNEL_NAMES

log = logging.getLogger(__name__)


def hdf5_to_mne(
    path: str | Path,
    channel_names: Optional[list[str]] = None,
    montage: str = "standard_1020",
) -> mne.io.RawArray:
    """Load an HDF5 session file and return an MNE RawArray.

    The HDF5 file stores EEG in microvolts (float32). MNE expects volts,
    so this function handles the unit conversion.

    Args:
        path: Path to the HDF5 session file.
        channel_names: Override channel names. If None, reads from file
            attributes or falls back to CHANNEL_NAMES.
        montage: MNE montage name for electrode positions.

    Returns:
        MNE RawArray with EEG data in volts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        # Read raw EEG data (n_channels, n_samples) in microvolts
        eeg_uv = f["raw"][:]
        sample_rate = int(f.attrs["sample_rate"])
        n_channels = int(f.attrs["n_channels"])

        # Resolve channel names
        if channel_names is None:
            if "channel_names" in f.attrs:
                stored = f.attrs["channel_names"]
                channel_names = [
                    s.decode("utf-8") if isinstance(s, bytes) else s
                    for s in stored
                ]
            elif n_channels == len(CHANNEL_NAMES):
                channel_names = list(CHANNEL_NAMES)
            else:
                channel_names = [f"CH{i + 1}" for i in range(n_channels)]

    # Convert microvolts -> volts for MNE
    eeg_v = eeg_uv * 1e-6

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sample_rate,
        ch_types="eeg",
    )

    raw = mne.io.RawArray(eeg_v, info, verbose=False)

    # Apply montage if all channel names are standard 10-20
    try:
        raw.set_montage(montage, on_missing="warn")
    except Exception:
        log.debug("Could not set montage %s; skipping", montage)

    log.info(
        "Loaded HDF5 -> MNE: %d channels, %d samples (%.1f s) @ %d Hz",
        n_channels, eeg_uv.shape[1],
        eeg_uv.shape[1] / sample_rate, sample_rate,
    )

    return raw


def mne_to_hdf5(
    raw: mne.io.BaseRaw,
    path: str | Path,
    chunk_samples: int = 1000,
) -> None:
    """Export an MNE Raw object to the realtime HDF5 session format.

    Useful for creating playback files from preprocessed data.

    Args:
        raw: MNE Raw object (data assumed to be in volts).
        path: Output HDF5 file path.
        chunk_samples: HDF5 chunk size.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_v = raw.get_data()  # (n_channels, n_samples) in volts
    data_uv = (data_v * 1e6).astype(np.float32)  # convert to microvolts
    n_channels, n_samples = data_uv.shape
    sfreq = int(raw.info["sfreq"])
    ch_names = raw.ch_names

    # Generate synthetic timestamps (evenly spaced)
    timestamps = np.arange(n_samples, dtype=np.float64) / sfreq

    with h5py.File(path, "w") as f:
        f.create_dataset(
            "raw", data=data_uv,
            chunks=(n_channels, min(chunk_samples, n_samples)),
            compression="gzip", compression_opts=1,
        )
        f.create_dataset("timestamps", data=timestamps)

        # Empty marker/feedback groups for schema compliance
        markers = f.create_group("markers")
        markers.create_dataset("sample_index", shape=(0,), maxshape=(None,), dtype=np.int64)
        markers.create_dataset("marker_id", shape=(0,), maxshape=(None,), dtype=np.int32)
        markers.create_dataset("label", shape=(0,), maxshape=(None,),
                               dtype=h5py.string_dtype(encoding="utf-8"))

        fb = f.create_group("feedback")
        fb.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
        fb.create_dataset("state", shape=(0,), maxshape=(None,), dtype=np.float32)
        fb.create_dataset("target", shape=(0,), maxshape=(None,), dtype=np.float32)

        f.attrs["sample_rate"] = sfreq
        f.attrs["n_channels"] = n_channels
        f.attrs["channel_names"] = np.array(ch_names, dtype="S32")
        f.attrs["duration_samples"] = n_samples

    log.info("Exported MNE -> HDF5: %s (%d ch, %d samples)", path, n_channels, n_samples)
