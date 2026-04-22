"""ICA-based artifact removal for scalp EEG recordings.

Workflow:
    raw = load_session("session.h5")
    ica = fit_ica(raw, n_components=15)
    labels = classify_components(raw, ica)
    ica.exclude = select_rejects(labels, confidence=0.7)
    raw_clean = apply_ica(raw, ica)

All functions are pure / don't mutate their inputs unless documented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import mne
import numpy as np
from mne.preprocessing import ICA

from ..convert import hdf5_to_mne, mne_to_hdf5

# Artifact types that should be rejected from the cleaned signal.
# Anything not in this set is treated as brain activity and kept.
REJECT_LABELS = {
    "eye blink", "muscle artifact", "heart beat", "line noise",
    "channel noise", "eye movement",
}


def load_session(path: Path | str) -> mne.io.Raw:
    """Load an HDF5 session into MNE Raw with annotations from markers.

    Wraps convert.hdf5_to_mne and adds marker annotations.
    """
    path = Path(path)
    raw = hdf5_to_mne(path)

    # Add marker annotations from the HDF5
    with h5py.File(path, "r") as f:
        if "markers" in f:
            m = f["markers"]
            si = m["sample_index"][:]
            if len(si) > 0:
                fs = int(f.attrs["sample_rate"])
                onsets = si.astype(np.float64) / fs
                durations = np.zeros_like(onsets)
                labels = [
                    s.decode() if isinstance(s, bytes) else s
                    for s in m["label"][:]
                ]
                raw.set_annotations(mne.Annotations(onsets, durations, labels))

    return raw


def fit_ica(
    raw: mne.io.Raw,
    n_components: int = 15,
    highpass_hz: float = 1.0,
    method: str = "picard",
    random_state: int = 42,
) -> ICA:
    """Fit ICA on a high-pass filtered copy.

    ICA assumes stationarity; slow drift (<1 Hz) violates this and
    contaminates the decomposition. We filter for fitting, but the
    resulting weights get applied to unfiltered data.

    Rule of thumb: n_components <= n_channels and <= n_good_channels.
    For a 16-channel montage with 4 railed channels, n_components=12
    is safer than 15.
    """
    raw_hp = raw.copy().filter(
        l_freq=highpass_hz, h_freq=None, verbose="ERROR"
    )
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw_hp, verbose="ERROR")
    return ica


def classify_components(raw: mne.io.Raw, ica: ICA) -> dict:
    """Run mne-icalabel on fitted components.

    Returns a dict with keys:
      - 'labels': list of strings per component (ICLabel taxonomy)
      - 'y_pred_proba': array of max probabilities per component

    ICLabel categories: 'brain', 'muscle artifact', 'eye blink',
    'heart beat', 'line noise', 'channel noise', 'other'
    """
    try:
        from mne_icalabel import label_components
    except ImportError as e:
        raise ImportError(
            "mne-icalabel is required. Install with: "
            "pip install -e .[analysis]"
        ) from e

    return label_components(raw, ica, method="iclabel")


def select_rejects(
    labels: dict,
    confidence: float = 0.7,
    reject_set: set[str] = REJECT_LABELS,
) -> list[int]:
    """Return component indices to exclude based on classification.

    Conservative: only exclude components where the classifier is
    >= confidence that the component is an artifact. Lower confidence
    keeps borderline components (safer — won't delete brain signal).
    """
    excludes = []
    for i, (lab, proba) in enumerate(
        zip(labels["labels"], labels["y_pred_proba"])
    ):
        if lab in reject_set and proba >= confidence:
            excludes.append(i)
    return excludes


def apply_ica(raw: mne.io.Raw, ica: ICA) -> mne.io.Raw:
    """Apply ICA exclusions to produce a cleaned copy. Does not mutate raw."""
    return ica.apply(raw.copy(), verbose="ERROR")


def save_cleaned(
    raw_clean: mne.io.Raw,
    out_path: Path | str,
    source_path: Path | str,
    ica_info: Optional[dict] = None,
) -> None:
    """Write a cleaned HDF5 preserving the original schema.

    Adds root attrs:
      - ica_applied: True
      - ica_source_file: original HDF5 filename
      - ica_n_excluded: number of components removed
      - ica_excluded_labels: comma-separated labels of excluded components
    """
    source_path = Path(source_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as src, h5py.File(out_path, "w") as dst:
        # Write cleaned EEG data (back to microvolts)
        cleaned_data = (raw_clean.get_data() * 1e6).astype(np.float32)
        dst.create_dataset(
            "raw", data=cleaned_data,
            chunks=(cleaned_data.shape[0], min(1000, cleaned_data.shape[1])),
            compression="gzip", compression_opts=1,
        )

        # Copy timestamps, markers, feedback from source
        if "timestamps" in src:
            dst.copy(src["timestamps"], "timestamps")
        if "markers" in src:
            dst.copy(src["markers"], "markers")
        if "feedback" in src:
            dst.copy(src["feedback"], "feedback")

        # Copy all source attrs
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        # Add ICA metadata
        dst.attrs["ica_applied"] = True
        dst.attrs["ica_source_file"] = str(source_path.name)
        if ica_info is not None:
            dst.attrs["ica_n_excluded"] = ica_info.get("n_excluded", 0)
            dst.attrs["ica_excluded_labels"] = ",".join(
                ica_info.get("excluded_labels", [])
            )
