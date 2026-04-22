"""ICA pipeline smoke tests. Real ICA validation happens with real recordings."""

import numpy as np
import pytest
import h5py

from openbci_eeg.realtime.analysis.ica import (
    load_session,
    fit_ica,
    apply_ica,
    select_rejects,
    save_cleaned,
    REJECT_LABELS,
)


def _make_synthetic_hdf5(path, n_channels=16, n_samples=30_000, fs=125):
    """Create a minimal HDF5 that load_session can read."""
    rng = np.random.default_rng(0)

    # 2-source synthetic: alpha on posterior channels, noise elsewhere
    t = np.arange(n_samples) / fs
    alpha = 20 * np.sin(2 * np.pi * 10 * t)
    data = rng.normal(0, 5, size=(n_channels, n_samples)).astype(np.float32)
    data[8:] += alpha  # inject alpha on channels 9-16

    ch_names = [
        "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "Oz",
        "C3", "C4", "Pz", "Cz", "P3", "P4", "O1", "O2",
    ]

    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=data)
        f.create_dataset("timestamps", data=t.astype(np.float64))
        f.attrs["channel_names"] = np.array(ch_names, dtype="S32")
        f.attrs["sample_rate"] = fs
        f.attrs["n_channels"] = n_channels
        f.attrs["montage_version"] = "test"
        grp = f.create_group("markers")
        grp.create_dataset(
            "sample_index", data=np.array([], dtype=np.int64)
        )
        grp.create_dataset(
            "marker_id", data=np.array([], dtype=np.int32)
        )
        grp.create_dataset(
            "label", data=np.array([], dtype=h5py.string_dtype())
        )


def test_load_session(tmp_path):
    p = tmp_path / "syn.h5"
    _make_synthetic_hdf5(p)
    raw = load_session(p)
    assert raw.info["nchan"] == 16
    assert raw.info["sfreq"] == 125
    # MNE stores in volts; data was 5uV noise + 20uV alpha
    assert np.abs(raw.get_data()).max() < 1e-3


def test_fit_ica_runs(tmp_path):
    p = tmp_path / "syn.h5"
    _make_synthetic_hdf5(p)
    raw = load_session(p)
    ica = fit_ica(raw, n_components=8)
    assert ica.n_components_ == 8


def test_apply_ica_preserves_shape(tmp_path):
    p = tmp_path / "syn.h5"
    _make_synthetic_hdf5(p)
    raw = load_session(p)
    ica = fit_ica(raw, n_components=8)
    ica.exclude = [0, 1]  # pretend to reject first two
    raw_clean = apply_ica(raw, ica)
    assert raw_clean.get_data().shape == raw.get_data().shape


def test_select_rejects_threshold():
    labels = {
        "labels": [
            "brain", "eye blink", "muscle artifact", "brain", "line noise",
        ],
        "y_pred_proba": np.array([0.95, 0.85, 0.50, 0.90, 0.99]),
    }
    # At confidence 0.7: eye blink (0.85) and line noise (0.99) pass;
    # muscle (0.50) doesn't
    assert select_rejects(labels, confidence=0.7) == [1, 4]
    # At confidence 0.4: all three artifacts pass
    assert select_rejects(labels, confidence=0.4) == [1, 2, 4]


def test_select_rejects_empty():
    labels = {
        "labels": ["brain", "brain"],
        "y_pred_proba": np.array([0.95, 0.90]),
    }
    assert select_rejects(labels, confidence=0.7) == []


def test_save_cleaned_schema(tmp_path):
    p = tmp_path / "syn.h5"
    _make_synthetic_hdf5(p)
    raw = load_session(p)
    ica = fit_ica(raw, n_components=8)
    ica.exclude = [0]
    raw_clean = apply_ica(raw, ica)

    out = tmp_path / "cleaned.h5"
    save_cleaned(
        raw_clean, out, p,
        ica_info={"n_excluded": 1, "excluded_labels": ["eye blink"]},
    )

    with h5py.File(out, "r") as f:
        assert "raw" in f
        assert f["raw"].shape == (16, 30_000)
        assert f.attrs["ica_applied"] == True
        assert f.attrs["ica_n_excluded"] == 1
        assert f.attrs["ica_excluded_labels"] == "eye blink"
        assert f.attrs["ica_source_file"] == "syn.h5"
        assert "timestamps" in f
        assert "markers" in f
