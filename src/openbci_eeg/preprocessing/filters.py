"""
EEG signal filtering: notch (line noise), bandpass, and combined preprocessing.

All functions operate on MNE Raw objects and return copies (non-destructive).
"""

from __future__ import annotations

import logging
from typing import Optional

import mne

from openbci_eeg.config import PreprocessConfig

logger = logging.getLogger(__name__)


def notch(
    raw: mne.io.RawArray,
    freq: float = 60.0,
    quality: float = 30.0,
) -> mne.io.RawArray:
    """
    Apply notch filter to remove line noise.

    Args:
        raw: MNE Raw object.
        freq: Line noise frequency (60 Hz US, 50 Hz EU).
        quality: Q factor for notch width.

    Returns:
        Filtered copy of Raw.
    """
    raw_notched = raw.copy().notch_filter(
        freqs=freq,
        quality=quality,
        verbose=False,
    )
    logger.debug("Notch filter applied at %.1f Hz (Q=%.1f)", freq, quality)
    return raw_notched


def bandpass(
    raw: mne.io.RawArray,
    l_freq: float = 0.5,
    h_freq: float = 50.0,
    method: str = "fir",
) -> mne.io.RawArray:
    """
    Apply bandpass filter.

    Args:
        raw: MNE Raw object.
        l_freq: Lower cutoff in Hz.
        h_freq: Upper cutoff in Hz. Capped at Nyquist - 1.
        method: "fir" or "iir".

    Returns:
        Filtered copy of Raw.
    """
    nyquist = raw.info["sfreq"] / 2
    if h_freq >= nyquist:
        h_freq = nyquist - 1
        logger.warning("h_freq capped to %.1f Hz (Nyquist limit)", h_freq)

    raw_bp = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        verbose=False,
    )
    logger.debug("Bandpass filter: %.1f - %.1f Hz (%s)", l_freq, h_freq, method)
    return raw_bp


def preprocess_raw(
    raw: mne.io.RawArray,
    config: Optional[PreprocessConfig] = None,
) -> mne.io.RawArray:
    """
    Full preprocessing pipeline: notch → bandpass.

    Args:
        raw: MNE Raw object.
        config: Preprocessing config. Uses defaults if None.

    Returns:
        Preprocessed copy of Raw.
    """
    if config is None:
        config = PreprocessConfig()

    result = notch(raw, freq=config.notch_freq, quality=config.notch_quality)
    result = bandpass(
        result,
        l_freq=config.bandpass_low,
        h_freq=config.bandpass_high,
        method=config.filter_method,
    )

    logger.info(
        "Preprocessing complete: notch=%.0f Hz, bandpass=%.1f-%.1f Hz",
        config.notch_freq,
        config.bandpass_low,
        config.bandpass_high,
    )
    return result
