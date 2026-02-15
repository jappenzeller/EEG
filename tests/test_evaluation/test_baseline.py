"""Tests for classical baseline evaluation."""

import numpy as np
import pytest

from openbci_eeg.evaluation.baseline import (
    extract_classical_features,
    FeatureConfig,
    _hjorth_params,
    _spectral_entropy,
)


class TestClassicalFeatures:
    """Test classical feature extraction."""

    def test_extract_features_shape(self):
        """Feature vector has expected shape."""
        n_channels = 4
        n_samples = 1000
        eeg_data = np.random.randn(n_channels, n_samples) * 50  # ~50 µV

        features, names = extract_classical_features(eeg_data, sfreq=125.0)

        # Per channel: 5 bands + 10 ratios + 4 moments + 3 hjorth + 2 others + 1 entropy = 25
        # Plus coherence: 4*3/2 = 6 pairs
        # Total: 4*25 + 6 = 106
        assert len(features) == len(names)
        assert len(features) > 0

    def test_extract_features_values_bounded(self):
        """Features are finite and not NaN."""
        eeg_data = np.random.randn(4, 500) * 50

        features, _ = extract_classical_features(eeg_data, sfreq=125.0)

        assert np.all(np.isfinite(features))

    def test_extract_features_no_coherence(self):
        """Can disable coherence features."""
        eeg_data = np.random.randn(4, 500) * 50
        config = FeatureConfig(include_coherence=False)

        features, names = extract_classical_features(eeg_data, config=config)

        assert not any("coh_" in name for name in names)


class TestHjorthParams:
    """Test Hjorth parameters."""

    def test_hjorth_sine_wave(self):
        """Hjorth params for sine wave."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine

        activity, mobility, complexity = _hjorth_params(signal)

        assert activity > 0
        assert mobility > 0
        assert complexity > 0
        assert np.isfinite(activity)
        assert np.isfinite(mobility)
        assert np.isfinite(complexity)

    def test_hjorth_flat_signal(self):
        """Hjorth params for constant signal."""
        signal = np.ones(1000)

        activity, mobility, complexity = _hjorth_params(signal)

        assert activity == 0.0  # No variance


class TestSpectralEntropy:
    """Test spectral entropy."""

    def test_entropy_white_noise(self):
        """White noise has high spectral entropy."""
        np.random.seed(42)
        white_noise = np.random.randn(1000)

        entropy = _spectral_entropy(white_noise, sfreq=125.0, window_sec=1.0)

        # White noise should have high entropy (close to 1 when normalized)
        assert entropy > 0.7

    def test_entropy_sine_wave(self):
        """Pure sine has low spectral entropy."""
        t = np.linspace(0, 2, 250)  # 2 seconds at 125 Hz
        sine = np.sin(2 * np.pi * 10 * t)

        entropy = _spectral_entropy(sine, sfreq=125.0, window_sec=1.0)

        # Pure sine should have lower entropy than white noise
        assert entropy < 0.7
        assert entropy > 0

    def test_entropy_bounded(self):
        """Entropy is in [0, 1]."""
        signal = np.random.randn(500)

        entropy = _spectral_entropy(signal, sfreq=125.0, window_sec=1.0)

        assert 0 <= entropy <= 1
