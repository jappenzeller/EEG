"""Tests for EEG preprocessing filters."""

import numpy as np
import pytest
import mne

from openbci_eeg.preprocessing.filters import notch, bandpass, preprocess_raw
from openbci_eeg.preprocessing.convert import brainflow_to_mne
from openbci_eeg.preprocessing.artifacts import detect_bad_channels


@pytest.fixture
def mne_raw(synthetic_eeg, channel_names, sample_rate):
    """Create MNE Raw from synthetic data (bypassing BrainFlow conversion)."""
    # Synthetic data is in µV, MNE wants V
    eeg_v = synthetic_eeg * 1e-6
    ch_types = ["eeg"] * len(channel_names)
    info = mne.create_info(ch_names=channel_names, sfreq=sample_rate, ch_types=ch_types)
    return mne.io.RawArray(eeg_v, info, verbose=False)


class TestNotchFilter:
    def test_output_shape(self, mne_raw):
        filtered = notch(mne_raw, freq=60.0)
        assert filtered.get_data().shape == mne_raw.get_data().shape

    def test_returns_copy(self, mne_raw):
        filtered = notch(mne_raw)
        assert filtered is not mne_raw


class TestBandpass:
    def test_output_shape(self, mne_raw):
        filtered = bandpass(mne_raw, l_freq=1.0, h_freq=50.0)
        assert filtered.get_data().shape == mne_raw.get_data().shape

    def test_nyquist_cap(self, mne_raw):
        # h_freq above Nyquist should be capped without error
        filtered = bandpass(mne_raw, l_freq=1.0, h_freq=100.0)
        assert filtered is not None


class TestPreprocessRaw:
    def test_full_pipeline(self, mne_raw):
        cleaned = preprocess_raw(mne_raw)
        assert cleaned.get_data().shape == mne_raw.get_data().shape


class TestBadChannelDetection:
    def test_no_bad_channels(self, mne_raw):
        bads = detect_bad_channels(mne_raw)
        # Synthetic data should have similar variance across channels
        assert len(bads) <= 2  # Allow some variance

    def test_detects_flat_channel(self, mne_raw):
        # Zero out one channel
        data = mne_raw.get_data()
        data[0, :] = 0
        raw_modified = mne.io.RawArray(data, mne_raw.info, verbose=False)
        bads = detect_bad_channels(raw_modified)
        assert mne_raw.ch_names[0] in bads
