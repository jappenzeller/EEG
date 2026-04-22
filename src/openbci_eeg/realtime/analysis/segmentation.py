"""Split windowed covariance sequences by paradigm state labels.

Uses marker annotations from the HDF5 (surfaced as MNE Annotations).
Returns dict[label -> cov_slice] for two-state analyses (e.g. polarity).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import mne


def segment_by_state(
    covs: np.ndarray,
    centers_sec: np.ndarray,
    raw: mne.io.BaseRaw,
    state_labels: Optional[Iterable[str]] = None,
) -> dict[str, np.ndarray]:
    """Split covariance array by annotation label.

    A window is assigned to a label if its center falls within the
    annotation interval for that label. Windows outside all annotated
    intervals are discarded.

    Markers from EyesMinimalRunner are point markers (duration 0).
    Each marker starts a state that runs until the next marker.

    Args:
        covs: (n_windows, n_channels, n_channels) from windowed_covariance
        centers_sec: (n_windows,) window center timestamps
        raw: Raw object with annotations set
        state_labels: labels to include; None returns all found

    Returns:
        dict mapping label -> (n_windows_in_state, n_channels, n_channels)
    """
    onsets = np.array([a["onset"] for a in raw.annotations])
    descs = [a["description"] for a in raw.annotations]

    skip_labels = {"session_start", "session_end"}
    intervals = []
    for i, (onset, label) in enumerate(zip(onsets, descs)):
        if label in skip_labels:
            continue
        end = onsets[i + 1] if i + 1 < len(onsets) else raw.times[-1]
        intervals.append((onset, end, label))

    wanted = set(state_labels) if state_labels is not None else None
    result: dict[str, list[np.ndarray]] = {}

    for onset, end, label in intervals:
        if wanted is not None and label not in wanted:
            continue
        mask = (centers_sec >= onset) & (centers_sec < end)
        if mask.any():
            result.setdefault(label, []).append(covs[mask])

    return {k: np.concatenate(v, axis=0) for k, v in result.items()}


def merge_states(
    segmented: dict[str, np.ndarray],
    groups: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Merge multiple state labels into grouped labels.

    Example:
        groups = {"open": ["eyes_open_1", "eyes_open_2"],
                  "closed": ["eyes_closed"]}
    """
    out = {}
    for group_name, member_labels in groups.items():
        arrs = [segmented[m] for m in member_labels if m in segmented]
        if arrs:
            out[group_name] = np.concatenate(arrs, axis=0)
    return out
