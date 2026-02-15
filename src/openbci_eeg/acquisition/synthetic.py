"""
Synthetic board for testing without hardware.

BrainFlow's SYNTHETIC_BOARD generates realistic EEG-like data with
alpha oscillations, noise, and proper channel structure. This module
provides convenience wrappers and optional signal injection for
controlled testing of the pipeline.
"""

from __future__ import annotations

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from openbci_eeg.acquisition.board import connect


def create_synthetic_board() -> BoardShim:
    """
    Create and connect a synthetic board for testing.

    Returns:
        Connected BoardShim using SYNTHETIC_BOARD.
    """
    return connect(synthetic=True)


def inject_alpha(
    data: np.ndarray,
    channels: list[int],
    freq: float = 10.0,
    amplitude_uv: float = 20.0,
    sfreq: float = 125.0,
) -> np.ndarray:
    """
    Inject alpha oscillation into specific channels of a data array.

    Useful for testing alpha detection / eyes-open-closed paradigms.

    Args:
        data: Full board data array (channels x samples).
        channels: Row indices to inject into.
        freq: Oscillation frequency in Hz.
        amplitude_uv: Peak amplitude in µV.
        sfreq: Sample rate.

    Returns:
        Modified data array (in-place and returned).
    """
    n_samples = data.shape[1]
    t = np.arange(n_samples) / sfreq
    signal = amplitude_uv * np.sin(2 * np.pi * freq * t)

    for ch in channels:
        data[ch, :] += signal

    return data


def inject_p300(
    data: np.ndarray,
    channels: list[int],
    onset_samples: list[int],
    amplitude_uv: float = 10.0,
    latency_sec: float = 0.3,
    width_sec: float = 0.1,
    sfreq: float = 125.0,
) -> np.ndarray:
    """
    Inject P300-like positive deflections at specified onset times.

    Args:
        data: Full board data array.
        channels: Row indices to inject into.
        onset_samples: Sample indices of stimulus onsets.
        amplitude_uv: P300 peak amplitude in µV.
        latency_sec: Time from onset to P300 peak.
        width_sec: Gaussian width of P300 component.
        sfreq: Sample rate.

    Returns:
        Modified data array.
    """
    n_samples = data.shape[1]
    latency_samp = int(latency_sec * sfreq)
    width_samp = int(width_sec * sfreq)

    for onset in onset_samples:
        peak = onset + latency_samp
        if peak >= n_samples:
            continue

        # Gaussian P300 waveform
        t_range = np.arange(max(0, peak - 4 * width_samp), min(n_samples, peak + 4 * width_samp))
        waveform = amplitude_uv * np.exp(-0.5 * ((t_range - peak) / width_samp) ** 2)

        for ch in channels:
            data[ch, t_range] += waveform

    return data


def inject_gamma_burst(
    data: np.ndarray,
    channels: list[int],
    start_sample: int,
    duration_samples: int,
    freq: float = 40.0,
    amplitude_uv: float = 5.0,
    sfreq: float = 125.0,
) -> np.ndarray:
    """
    Inject a burst of gamma oscillation into specific channels.

    Args:
        data: Full board data array.
        channels: Row indices to inject into.
        start_sample: Start of gamma burst.
        duration_samples: Length of burst.
        freq: Gamma frequency in Hz.
        amplitude_uv: Amplitude in µV.
        sfreq: Sample rate.

    Returns:
        Modified data array.
    """
    end = min(start_sample + duration_samples, data.shape[1])
    t = np.arange(end - start_sample) / sfreq

    # Tapered gamma burst (Hann envelope)
    envelope = np.hanning(len(t))
    signal = amplitude_uv * envelope * np.sin(2 * np.pi * freq * t)

    for ch in channels:
        data[ch, start_sample:end] += signal

    return data
