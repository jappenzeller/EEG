"""
Shared test fixtures for openbci-eeg test suite.

Uses BrainFlow synthetic board for all tests that need EEG data,
so no hardware is required to run the test suite.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_rate():
    """Standard sample rate for Cyton+Daisy."""
    return 125.0


@pytest.fixture
def n_channels():
    """Standard channel count."""
    return 16


@pytest.fixture
def channel_names():
    """Standard 10-20 channel names."""
    from openbci_eeg import CHANNEL_NAMES
    return list(CHANNEL_NAMES)


@pytest.fixture
def synthetic_eeg(sample_rate, n_channels):
    """
    Generate 10 seconds of synthetic EEG data.

    Includes:
    - 10 Hz alpha oscillation (20 µV) on all channels
    - Pink noise background
    - Slightly different amplitudes per channel
    """
    rng = np.random.default_rng(42)
    duration_sec = 10.0
    n_samples = int(duration_sec * sample_rate)

    # Pink noise (1/f)
    freqs = np.fft.rfftfreq(n_samples, d=1 / sample_rate)
    freqs[0] = 1  # Avoid division by zero
    pink_spectrum = 1 / np.sqrt(freqs)

    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Random phase pink noise
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        spectrum = pink_spectrum * np.exp(1j * phases)
        noise = np.fft.irfft(spectrum, n=n_samples) * 10  # Scale to ~10 µV

        # Add alpha
        t = np.arange(n_samples) / sample_rate
        alpha_amp = 15 + rng.normal(0, 3)  # Vary per channel
        alpha = alpha_amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))

        data[ch] = noise + alpha

    return data


@pytest.fixture
def synthetic_eeg_single_channel(synthetic_eeg):
    """Single channel of synthetic EEG data."""
    return synthetic_eeg[0]


@pytest.fixture
def default_config():
    """Default pipeline configuration."""
    from openbci_eeg.config import PipelineConfig
    return PipelineConfig()


@pytest.fixture
def pn_config():
    """Default PN configuration."""
    from openbci_eeg.config import PNConfig
    return PNConfig()


@pytest.fixture
def sample_pn_params(synthetic_eeg, channel_names, sample_rate, pn_config):
    """Pre-computed PN parameters for testing."""
    from openbci_eeg.pn_extraction import extract_pn_multichannel
    return extract_pn_multichannel(synthetic_eeg, sample_rate, channel_names, pn_config)
