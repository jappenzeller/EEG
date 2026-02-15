"""Tests for PN parameter extraction."""

import numpy as np
import pytest

from openbci_eeg.config import PNConfig
from openbci_eeg.pn_extraction.dynamics import extract_pn_single, extract_pn_multichannel
from openbci_eeg.pn_extraction.envelope import rms_envelope, normalize_envelope


class TestRMSEnvelope:
    def test_output_shape(self, synthetic_eeg_single_channel, sample_rate):
        rms = rms_envelope(synthetic_eeg_single_channel, sample_rate)
        assert rms.shape == synthetic_eeg_single_channel.shape

    def test_non_negative(self, synthetic_eeg_single_channel, sample_rate):
        rms = rms_envelope(synthetic_eeg_single_channel, sample_rate)
        assert np.all(rms >= 0)

    def test_zero_signal(self, sample_rate):
        signal = np.zeros(1000)
        rms = rms_envelope(signal, sample_rate)
        assert np.allclose(rms, 0)


class TestNormalizeEnvelope:
    def test_minmax_range(self):
        env = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_envelope(env, method="minmax")
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

    def test_sigmoid_range(self):
        env = np.random.randn(1000)
        normed = normalize_envelope(env, method="zscore")
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            normalize_envelope(np.ones(10), method="bad")


class TestExtractPNSingle:
    def test_output_keys(self, synthetic_eeg_single_channel, sample_rate):
        result = extract_pn_single(synthetic_eeg_single_channel, sample_rate)
        assert set(result.keys()) == {"a", "b", "c"}

    def test_output_shapes(self, synthetic_eeg_single_channel, sample_rate):
        n = len(synthetic_eeg_single_channel)
        result = extract_pn_single(synthetic_eeg_single_channel, sample_rate)
        assert result["a"].shape == (n,)
        assert result["b"].shape == (n,)
        assert result["c"].shape == (n,)

    def test_a_bounded(self, synthetic_eeg_single_channel, sample_rate):
        result = extract_pn_single(synthetic_eeg_single_channel, sample_rate)
        assert np.all(result["a"] >= 0) and np.all(result["a"] <= 1)

    def test_c_bounded(self, synthetic_eeg_single_channel, sample_rate):
        result = extract_pn_single(synthetic_eeg_single_channel, sample_rate)
        assert np.all(result["c"] >= 0) and np.all(result["c"] <= 1)

    def test_b_phase_range(self, synthetic_eeg_single_channel, sample_rate):
        result = extract_pn_single(synthetic_eeg_single_channel, sample_rate)
        assert np.all(result["b"] >= 0) and np.all(result["b"] <= 2 * np.pi)

    def test_rk4_vs_euler(self, synthetic_eeg_single_channel, sample_rate):
        config_euler = PNConfig(solver="euler")
        config_rk4 = PNConfig(solver="rk4")
        r_euler = extract_pn_single(synthetic_eeg_single_channel, sample_rate, config_euler)
        r_rk4 = extract_pn_single(synthetic_eeg_single_channel, sample_rate, config_rk4)
        # Should be similar but not identical
        assert not np.array_equal(r_euler["a"], r_rk4["a"])
        assert np.corrcoef(r_euler["a"], r_rk4["a"])[0, 1] > 0.95


class TestExtractPNMultichannel:
    def test_all_channels_present(self, synthetic_eeg, channel_names, sample_rate):
        result = extract_pn_multichannel(synthetic_eeg, sample_rate, channel_names)
        assert set(result.keys()) == set(channel_names)

    def test_per_channel_structure(self, synthetic_eeg, channel_names, sample_rate):
        result = extract_pn_multichannel(synthetic_eeg, sample_rate, channel_names)
        for ch in channel_names:
            assert set(result[ch].keys()) == {"a", "b", "c"}
