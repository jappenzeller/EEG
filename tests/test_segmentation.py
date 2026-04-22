"""Segmentation module tests."""

import numpy as np
import mne

from openbci_eeg.realtime.analysis.segmentation import (
    segment_by_state,
    merge_states,
)


def _make_raw_with_annotations():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, size=(16, 30_000))
    info = mne.create_info(
        [f"CH{i+1}" for i in range(16)], 125, ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    annotations = mne.Annotations(
        onset=[60.0, 120.0, 180.0],
        duration=[0.0, 0.0, 0.0],
        description=["eyes_open_1", "eyes_closed", "eyes_open_2"],
    )
    raw.set_annotations(annotations)
    return raw


def test_segment_splits_correctly():
    raw = _make_raw_with_annotations()
    n_windows = 120
    covs = np.array([np.eye(16) for _ in range(n_windows)])
    centers = np.arange(0, n_windows) * 2.0 + 1.0
    out = segment_by_state(covs, centers, raw)
    assert "eyes_open_1" in out
    assert "eyes_closed" in out
    assert "eyes_open_2" in out
    assert out["eyes_open_1"].shape[0] == 30
    assert out["eyes_closed"].shape[0] == 30


def test_merge_groups_two_opens():
    raw = _make_raw_with_annotations()
    covs = np.array([np.eye(16) for _ in range(120)])
    centers = np.arange(0, 120) * 2.0 + 1.0
    segmented = segment_by_state(covs, centers, raw)
    merged = merge_states(
        segmented,
        groups={
            "open": ["eyes_open_1", "eyes_open_2"],
            "closed": ["eyes_closed"],
        },
    )
    assert "open" in merged and "closed" in merged
    assert merged["open"].shape[0] == (
        segmented["eyes_open_1"].shape[0]
        + segmented["eyes_open_2"].shape[0]
    )
