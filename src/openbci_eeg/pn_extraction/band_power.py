"""
V4 Multi-Scale Band Power encoding for PN parameters.

Maps EEG frequency bands to PN parameters based on neurophysiological correspondence:
    a: (beta + gamma) relative power — excitatory proxy
    c: (delta + theta) relative power — inhibitory proxy
    b: alpha coherence across channels — thalamocortical coupling proxy

This differs from V1-V3 encodings by using frequency-domain features rather than
time-domain ODE dynamics. The band-to-parameter mapping follows the neurophysiology:
fast rhythms (beta/gamma) reflect excitatory cortical activity, slow rhythms
(delta/theta) reflect inhibitory/sleep dynamics, and alpha coherence reflects
thalamocortical gating.

Note: 125 Hz sample rate limits Nyquist to 62.5 Hz. High gamma (60-100 Hz) is
unavailable. Gamma band is defined as 35-50 Hz for this hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import welch, coherence as scipy_coherence

logger = logging.getLogger(__name__)


@dataclass
class BandConfig:
    """Frequency band definitions (Hz). Adjusted for 125 Hz sample rate."""
    delta: tuple[float, float] = (0.5, 4.0)
    theta: tuple[float, float] = (4.0, 8.0)
    alpha: tuple[float, float] = (8.0, 13.0)
    beta: tuple[float, float] = (13.0, 30.0)
    gamma: tuple[float, float] = (30.0, 50.0)  # Capped at 50 Hz (Nyquist - margin)


def compute_band_powers(
    eeg_channel: np.ndarray,
    sfreq: float = 125.0,
    window_sec: float = 2.0,
    bands: Optional[BandConfig] = None,
) -> dict[str, float]:
    """
    Compute relative power in each frequency band for a single channel.

    Args:
        eeg_channel: 1D array of EEG data (in µV).
        sfreq: Sample rate in Hz.
        window_sec: Window length for Welch PSD estimation.
        bands: Frequency band definitions.

    Returns:
        Dict with keys 'delta', 'theta', 'alpha', 'beta', 'gamma',
        each a relative power value in [0, 1] summing to ~1.
    """
    if bands is None:
        bands = BandConfig()

    nperseg = int(window_sec * sfreq)
    nperseg = min(nperseg, len(eeg_channel))

    freqs, psd = welch(eeg_channel, fs=sfreq, nperseg=nperseg)

    total_power = np.trapezoid(psd, freqs)
    if total_power == 0:
        total_power = 1e-10  # Avoid division by zero

    def band_power(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return float(np.trapezoid(psd[mask], freqs[mask]) / total_power)

    return {
        "delta": band_power(*bands.delta),
        "theta": band_power(*bands.theta),
        "alpha": band_power(*bands.alpha),
        "beta": band_power(*bands.beta),
        "gamma": band_power(*bands.gamma),
    }


def compute_alpha_coherence(
    eeg_data: np.ndarray,
    sfreq: float = 125.0,
    window_sec: float = 2.0,
    alpha_band: tuple[float, float] = (8.0, 13.0),
) -> float:
    """
    Compute mean pairwise alpha-band coherence across all channels.

    This serves as the 'b' parameter for V4 encoding, representing
    thalamocortical coupling strength.

    Args:
        eeg_data: 2D array of shape (n_channels, n_samples).
        sfreq: Sample rate in Hz.
        window_sec: Window length for coherence estimation.
        alpha_band: Alpha frequency range.

    Returns:
        Mean coherence value in [0, 1].
    """
    n_channels = eeg_data.shape[0]
    if n_channels < 2:
        return 0.5  # Default for single channel

    nperseg = int(window_sec * sfreq)
    nperseg = min(nperseg, eeg_data.shape[1])

    coherence_values = []

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, coh = scipy_coherence(
                eeg_data[i], eeg_data[j], fs=sfreq, nperseg=nperseg
            )
            # Mean coherence in alpha band
            mask = (freqs >= alpha_band[0]) & (freqs < alpha_band[1])
            if np.any(mask):
                coherence_values.append(np.mean(coh[mask]))

    if not coherence_values:
        return 0.5

    return float(np.mean(coherence_values))


def extract_v4_single(
    eeg_channel: np.ndarray,
    sfreq: float = 125.0,
    window_sec: float = 2.0,
    bands: Optional[BandConfig] = None,
) -> dict[str, float]:
    """
    Extract V4 encoding parameters for a single channel.

    Note: For single-channel extraction, 'b' (coherence) cannot be computed
    and defaults to the channel's alpha relative power as a proxy.

    Args:
        eeg_channel: 1D array of EEG data.
        sfreq: Sample rate in Hz.
        window_sec: Window length for spectral estimation.
        bands: Frequency band definitions.

    Returns:
        Dict with 'a', 'b', 'c' parameters:
            a: beta + gamma relative power (excitatory)
            b: alpha relative power (single-channel proxy for coherence)
            c: delta + theta relative power (inhibitory)
    """
    bp = compute_band_powers(eeg_channel, sfreq, window_sec, bands)

    a = bp["beta"] + bp["gamma"]
    c = bp["delta"] + bp["theta"]
    b = bp["alpha"]  # Single-channel fallback

    # Normalize a and c to [0, 1] (they already sum to <= 1 by construction)
    # Map b (alpha power) to [0, 2π] for Rz gate
    b_scaled = b * 2 * np.pi

    return {"a": float(a), "b": float(b_scaled), "c": float(c)}


def extract_v4_multichannel(
    eeg_data: np.ndarray,
    sfreq: float = 125.0,
    channel_names: Optional[list[str]] = None,
    window_sec: float = 2.0,
    bands: Optional[BandConfig] = None,
    coherence_mode: str = "global",
) -> dict[str, dict[str, float]]:
    """
    Extract V4 encoding parameters for all channels.

    Args:
        eeg_data: 2D array of shape (n_channels, n_samples).
        sfreq: Sample rate in Hz.
        channel_names: Names for each channel.
        window_sec: Window length for spectral estimation.
        bands: Frequency band definitions.
        coherence_mode: How to compute 'b' parameter:
            - "global": Same alpha coherence for all channels (mean pairwise)
            - "per_channel": Alpha power per channel (no cross-channel info)

    Returns:
        Dict keyed by channel name, each containing {'a', 'b', 'c'}.
    """
    n_channels = eeg_data.shape[0]

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    if bands is None:
        bands = BandConfig()

    # Compute global alpha coherence if using global mode
    if coherence_mode == "global":
        global_coherence = compute_alpha_coherence(eeg_data, sfreq, window_sec)
        b_scaled = global_coherence * 2 * np.pi
    else:
        global_coherence = None
        b_scaled = None

    results = {}
    for i, name in enumerate(channel_names):
        bp = compute_band_powers(eeg_data[i], sfreq, window_sec, bands)

        a = bp["beta"] + bp["gamma"]
        c = bp["delta"] + bp["theta"]

        if coherence_mode == "global":
            b = b_scaled
        else:
            # Per-channel: use alpha power as proxy
            b = bp["alpha"] * 2 * np.pi

        results[name] = {"a": float(a), "b": float(b), "c": float(c)}

    logger.info(
        "V4 extraction complete for %d channels (coherence_mode=%s)",
        n_channels,
        coherence_mode,
    )
    return results


def extract_v4_windowed(
    eeg_data: np.ndarray,
    sfreq: float = 125.0,
    channel_names: Optional[list[str]] = None,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    bands: Optional[BandConfig] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Extract V4 parameters over sliding windows, producing time series.

    Args:
        eeg_data: 2D array of shape (n_channels, n_samples).
        sfreq: Sample rate in Hz.
        channel_names: Names for each channel.
        window_sec: Window length for spectral estimation.
        step_sec: Step size between windows.
        bands: Frequency band definitions.

    Returns:
        Dict keyed by channel name, each containing:
            {'a': array, 'b': array, 'c': array} of shape (n_windows,)
    """
    n_channels, n_samples = eeg_data.shape

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    if bands is None:
        bands = BandConfig()

    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    n_windows = (n_samples - window_samples) // step_samples + 1

    if n_windows <= 0:
        raise ValueError(
            f"Signal too short ({n_samples} samples) for window_sec={window_sec}"
        )

    # Initialize output arrays
    results = {
        name: {"a": np.zeros(n_windows), "b": np.zeros(n_windows), "c": np.zeros(n_windows)}
        for name in channel_names
    }

    for w in range(n_windows):
        start = w * step_samples
        end = start + window_samples
        window_data = eeg_data[:, start:end]

        # Extract V4 for this window
        window_pn = extract_v4_multichannel(
            window_data, sfreq, channel_names, window_sec, bands, coherence_mode="global"
        )

        for name in channel_names:
            results[name]["a"][w] = window_pn[name]["a"]
            results[name]["b"][w] = window_pn[name]["b"]
            results[name]["c"][w] = window_pn[name]["c"]

    logger.info("V4 windowed extraction: %d windows of %.1fs", n_windows, window_sec)
    return results
