"""SPD covariance module smoke + property tests."""

import numpy as np
import pytest
import mne

from openbci_eeg.realtime.analysis.covariance import (
    windowed_covariance,
    is_spd,
    _cov_estimate,
)


def _make_synthetic_raw(n_channels=16, n_samples=30_000, fs=125):
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, size=(n_channels, n_samples))
    ch_names = [f"CH{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names, fs, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_all_covariances_are_spd():
    raw = _make_synthetic_raw()
    covs, _ = windowed_covariance(raw, window_sec=2.0)
    assert covs.ndim == 3
    for c in covs:
        assert is_spd(c)


def test_window_count_matches_expected():
    raw = _make_synthetic_raw(n_samples=12_500, fs=125)  # 100 seconds
    covs, centers = windowed_covariance(raw, window_sec=2.0, overlap=0.0)
    assert covs.shape[0] == 50
    assert centers[0] == pytest.approx(1.0)
    assert centers[-1] == pytest.approx(99.0)


def test_overlap_increases_window_count():
    raw = _make_synthetic_raw(n_samples=12_500, fs=125)
    covs_none, _ = windowed_covariance(raw, window_sec=2.0, overlap=0.0)
    covs_half, _ = windowed_covariance(raw, window_sec=2.0, overlap=0.5)
    assert covs_half.shape[0] > covs_none.shape[0]


def test_window_too_short_rejected():
    raw = _make_synthetic_raw(n_channels=16, n_samples=1000, fs=125)
    with pytest.raises(ValueError, match="too short"):
        windowed_covariance(raw, window_sec=0.1)


def test_dtype_is_float64():
    raw = _make_synthetic_raw()
    covs, _ = windowed_covariance(raw, window_sec=2.0)
    assert covs.dtype == np.float64


def test_cov_estimate_methods_all_spd():
    rng = np.random.default_rng(1)
    window = rng.normal(0, 1, size=(16, 250))
    for method in ["oas", "lw", "empirical"]:
        cov = _cov_estimate(window, method)
        assert is_spd(cov), f"{method} produced non-SPD result"
