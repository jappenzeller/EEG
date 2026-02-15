"""
PN neuron dynamics: solve excitatory (a) and inhibitory (c) ODEs from EEG signal.

The PN model encodes EEG channel activity as a 3-tuple (a, b, c):
    a: excitatory state [0, 1] — driven by signal energy
    b: shared phase [0, 2π] — instantaneous phase via Hilbert transform
    c: inhibitory state [0, 1] — accumulates with signal, slower decay

These map directly to the QDNU A-Gate quantum circuit parameters.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import hilbert

from openbci_eeg.config import PNConfig
from openbci_eeg.pn_extraction.envelope import rms_envelope, normalize_envelope

logger = logging.getLogger(__name__)


def extract_pn_single(
    eeg_channel: np.ndarray,
    sfreq: float = 125.0,
    config: Optional[PNConfig] = None,
) -> dict[str, np.ndarray]:
    """
    Extract PN parameters from a single EEG channel.

    Args:
        eeg_channel: 1D array of EEG data (in µV).
        sfreq: Sample rate in Hz.
        config: PN model parameters. Uses defaults if None.

    Returns:
        Dict with keys 'a', 'b', 'c', each a 1D array matching input length.
            a: excitatory state [0, 1]
            b: phase [0, 2π]
            c: inhibitory state [0, 1]
    """
    if config is None:
        config = PNConfig()

    # --- Driving function f(t): normalized RMS envelope ---
    rms = rms_envelope(eeg_channel, sfreq, window_sec=config.rms_window_sec)
    f = normalize_envelope(rms)

    # --- Phase via Hilbert transform ---
    analytic = hilbert(eeg_channel)
    b = np.angle(analytic) % (2 * np.pi)

    # --- Solve PN ODEs ---
    if config.solver == "rk4":
        a, c = _solve_pn_rk4(f, sfreq, config)
    else:
        a, c = _solve_pn_euler(f, sfreq, config)

    return {"a": a, "b": b, "c": c}


def extract_pn_multichannel(
    eeg_data: np.ndarray,
    sfreq: float = 125.0,
    channel_names: Optional[list[str]] = None,
    config: Optional[PNConfig] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Extract PN parameters for all channels.

    Args:
        eeg_data: 2D array of shape (n_channels, n_samples) in µV.
        sfreq: Sample rate in Hz.
        channel_names: Optional names for each channel.
        config: PN model parameters.

    Returns:
        Dict keyed by channel name (or index), each containing
        {'a': array, 'b': array, 'c': array}.
    """
    n_channels = eeg_data.shape[0]

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    results = {}
    for i, name in enumerate(channel_names):
        results[name] = extract_pn_single(eeg_data[i], sfreq, config)

    logger.info("PN extraction complete for %d channels.", n_channels)
    return results


# ---------------------------------------------------------------------------
# ODE Solvers
# ---------------------------------------------------------------------------

def _solve_pn_euler(
    f: np.ndarray,
    sfreq: float,
    config: PNConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler method for PN ODEs. Fast, first-order."""
    dt = 1.0 / sfreq
    n = len(f)

    a = np.zeros(n)
    c = np.zeros(n)
    a[0] = config.initial_a
    c[0] = config.initial_c

    la = config.lambda_a
    lc = config.lambda_c

    for i in range(1, n):
        a[i] = a[i - 1] + dt * (-la * a[i - 1] + f[i - 1] * (1.0 - a[i - 1]))
        c[i] = c[i - 1] + dt * (lc * c[i - 1] + f[i - 1] * (1.0 - c[i - 1]))

    # Clamp to [0, 1]
    np.clip(a, 0.0, 1.0, out=a)
    np.clip(c, 0.0, 1.0, out=c)

    return a, c


def _solve_pn_rk4(
    f: np.ndarray,
    sfreq: float,
    config: PNConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """4th-order Runge-Kutta for PN ODEs. More accurate, ~4x slower."""
    dt = 1.0 / sfreq
    n = len(f)

    a = np.zeros(n)
    c = np.zeros(n)
    a[0] = config.initial_a
    c[0] = config.initial_c

    la = config.lambda_a
    lc = config.lambda_c

    def da(a_val: float, f_val: float) -> float:
        return -la * a_val + f_val * (1.0 - a_val)

    def dc(c_val: float, f_val: float) -> float:
        return lc * c_val + f_val * (1.0 - c_val)

    for i in range(1, n):
        # Interpolate f between steps
        f0 = f[i - 1]
        f1 = f[i]
        fmid = 0.5 * (f0 + f1)

        # RK4 for a
        k1a = dt * da(a[i - 1], f0)
        k2a = dt * da(a[i - 1] + 0.5 * k1a, fmid)
        k3a = dt * da(a[i - 1] + 0.5 * k2a, fmid)
        k4a = dt * da(a[i - 1] + k3a, f1)
        a[i] = a[i - 1] + (k1a + 2 * k2a + 2 * k3a + k4a) / 6

        # RK4 for c
        k1c = dt * dc(c[i - 1], f0)
        k2c = dt * dc(c[i - 1] + 0.5 * k1c, fmid)
        k3c = dt * dc(c[i - 1] + 0.5 * k2c, fmid)
        k4c = dt * dc(c[i - 1] + k3c, f1)
        c[i] = c[i - 1] + (k1c + 2 * k2c + 2 * k3c + k4c) / 6

    np.clip(a, 0.0, 1.0, out=a)
    np.clip(c, 0.0, 1.0, out=c)

    return a, c
