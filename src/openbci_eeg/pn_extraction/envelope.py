"""
Signal envelope extraction for PN driving function f(t).

The driving function f(t) is a normalized RMS envelope of the EEG signal,
mapping raw voltage fluctuations to a [0, 1] activation level.
"""

from __future__ import annotations

import numpy as np


def rms_envelope(
    signal: np.ndarray,
    sfreq: float = 125.0,
    window_sec: float = 0.1,
) -> np.ndarray:
    """
    Compute RMS (root-mean-square) envelope of signal.

    Uses a sliding window convolution for efficiency.

    Args:
        signal: 1D array of EEG data.
        sfreq: Sample rate in Hz.
        window_sec: Window length in seconds.

    Returns:
        1D RMS envelope array, same length as input.
    """
    window_samples = max(1, int(window_sec * sfreq))
    kernel = np.ones(window_samples) / window_samples
    rms = np.sqrt(np.convolve(signal ** 2, kernel, mode="same"))
    return rms


def normalize_envelope(
    envelope: np.ndarray,
    method: str = "minmax",
    epsilon: float = 1e-10,
) -> np.ndarray:
    """
    Normalize envelope to [0, 1] range.

    Args:
        envelope: 1D RMS envelope array.
        method: "minmax" (default) or "zscore" (then sigmoid).
        epsilon: Small value to avoid division by zero.

    Returns:
        Normalized envelope in [0, 1].
    """
    if method == "minmax":
        emin = envelope.min()
        emax = envelope.max()
        denom = emax - emin + epsilon
        return (envelope - emin) / denom

    elif method == "zscore":
        # Z-score then sigmoid to [0, 1]
        mu = envelope.mean()
        sigma = envelope.std() + epsilon
        z = (envelope - mu) / sigma
        return 1.0 / (1.0 + np.exp(-z))

    else:
        raise ValueError(f"Unknown normalization method: {method}")
