"""Triptych smoke tests on synthetic data."""

import numpy as np
import h5py
import pytest

from openbci_eeg.realtime.analysis.ica import hdf5_to_mne as load_hdf5_to_mne
from openbci_eeg.realtime.analysis.triptych import (
    render_triptych,
    _band_power_per_channel,
    _default_psd_channels,
    _compute_psd,
)


def _make_synthetic_hdf5(path, n_channels=16, n_samples=30_000, fs=125):
    rng = np.random.default_rng(0)
    t = np.arange(n_samples) / fs
    data = rng.normal(0, 5, size=(n_channels, n_samples)).astype(np.float32)
    # Inject 10 Hz alpha on last 4 channels (P3, P4, O1, O2)
    data[-4:] += (15 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

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
        grp = f.create_group("markers")
        grp.create_dataset(
            "sample_index",
            data=np.array([0, 12500, 25000], dtype=np.int64),
        )
        grp.create_dataset(
            "marker_id", data=np.array([151, 152, 151], dtype=np.int32),
        )
        grp.create_dataset(
            "label",
            data=np.array(
                ["eyes_open_1", "eyes_closed", "eyes_open_2"],
                dtype=h5py.string_dtype(),
            ),
        )


def test_band_power_finds_alpha_on_occipital(tmp_path):
    p = tmp_path / "syn.h5"
    _make_synthetic_hdf5(p)
    raw = load_hdf5_to_mne(p)
    powers = _band_power_per_channel(raw, band=(8.0, 12.0))
    # Alpha injected on last 4 array rows = P3, P4, O1, O2
    occipital = ["P3", "P4", "O1", "O2"]
    occ_powers = [powers[ch] for ch in occipital]
    other_powers = [powers[ch] for ch in raw.ch_names if ch not in occipital]
    assert min(occ_powers) > max(other_powers)


def test_default_psd_channels_prefers_occipital():
    chs = ["Fp1", "Fp2", "F3", "F4", "Cz", "Pz", "O1", "O2", "Oz"]
    chosen = _default_psd_channels(chs)
    assert "Oz" in chosen
    assert len(chosen) == 3


def test_default_psd_channels_falls_back():
    chs = ["X1", "X2", "X3", "X4"]
    chosen = _default_psd_channels(chs)
    assert chosen == ["X1", "X2", "X3"]


def test_compute_psd_returns_expected_shape(tmp_path):
    p = tmp_path / "syn.h5"
    _make_synthetic_hdf5(p)
    raw = load_hdf5_to_mne(p)
    out = _compute_psd(raw, ["O1", "O2"])
    assert set(out.keys()) == {"O1", "O2"}
    for freqs, psd in out.values():
        assert freqs.ndim == 1
        assert freqs.shape == psd.shape


def test_render_triptych_runs_end_to_end(tmp_path):
    """Smoke test: full pipeline produces a figure without crashing."""
    p = tmp_path / "syn.h5"
    _make_synthetic_hdf5(p)
    raw = load_hdf5_to_mne(p)
    out_png = tmp_path / "triptych.png"
    result = render_triptych(
        raw, tmin=10.0, tmax=50.0, band=(8.0, 12.0),
        output_path=out_png,
    )
    assert result.figure is not None
    assert out_png.exists()
    assert len(result.band_powers) == raw.info["nchan"]
    assert len(result.psd_data) >= 1


def test_render_triptych_handles_too_few_positioned_channels(tmp_path):
    """Non-standard channel names should still render with source unavailable."""
    p = tmp_path / "weird.h5"
    rng = np.random.default_rng(0)
    n, fs, samples = 4, 125, 5000
    data = rng.normal(0, 5, size=(n, samples)).astype(np.float32)
    with h5py.File(p, "w") as f:
        f.create_dataset("raw", data=data)
        f.create_dataset("timestamps", data=np.arange(samples) / fs)
        f.attrs["channel_names"] = np.array(
            ["A", "B", "C", "D"], dtype="S32",
        )
        f.attrs["sample_rate"] = fs
        f.attrs["n_channels"] = n
        grp = f.create_group("markers")
        grp.create_dataset(
            "sample_index", data=np.array([], dtype=np.int64),
        )
        grp.create_dataset(
            "marker_id", data=np.array([], dtype=np.int32),
        )
        grp.create_dataset(
            "label", data=np.array([], dtype=h5py.string_dtype()),
        )

    raw = load_hdf5_to_mne(p)
    out_png = tmp_path / "weird_triptych.png"
    result = render_triptych(
        raw, tmin=0.0, tmax=20.0, band=(8.0, 12.0),
        output_path=out_png,
    )
    assert result.source_estimate is None
    assert out_png.exists()
