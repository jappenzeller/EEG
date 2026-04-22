"""SPD covariance matrices from multichannel EEG windows.

Produces the canonical QNFM input: one (n_channels x n_channels) SPD
matrix per time window, regularized to guarantee positive-definiteness
for downstream Riemannian / SPD-manifold methods.

Design choices:
    - Non-overlapping windows by default (overlap=0.0).
    - Shrinkage regularization via OAS (default) or Ledoit-Wolf.
    - Output dtype float64 for numerical stability.
"""

from __future__ import annotations

import numpy as np
import mne


def windowed_covariance(
    raw: mne.io.BaseRaw,
    window_sec: float = 2.0,
    overlap: float = 0.0,
    method: str = "oas",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one SPD covariance matrix per window.

    Args:
        raw: MNE Raw object (volts, MNE convention)
        window_sec: window length in seconds
        overlap: fractional overlap in [0, 1)
        method: 'oas' (Oracle Approximating Shrinkage), 'lw' (Ledoit-Wolf),
                or 'empirical'

    Returns:
        covs: (n_windows, n_channels, n_channels) float64 SPD matrices
        centers_sec: (n_windows,) float64 window center timestamps
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if method not in {"oas", "lw", "empirical"}:
        raise ValueError(f"unknown method: {method}")

    data = raw.get_data()
    fs = raw.info["sfreq"]
    n_channels, n_samples = data.shape

    window_samples = int(round(window_sec * fs))
    if window_samples < 2 * n_channels:
        raise ValueError(
            f"window ({window_samples} samples) too short for {n_channels} "
            f"channels. Need at least 2*n_channels samples for stable "
            f"covariance."
        )
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap))))

    starts = np.arange(0, n_samples - window_samples + 1, hop_samples)
    n_windows = len(starts)
    covs = np.zeros((n_windows, n_channels, n_channels), dtype=np.float64)
    centers_sec = np.zeros(n_windows, dtype=np.float64)

    for i, s in enumerate(starts):
        window = data[:, s:s + window_samples].astype(np.float64)
        covs[i] = _cov_estimate(window, method)
        centers_sec[i] = (s + window_samples / 2.0) / fs

    return covs, centers_sec


def _cov_estimate(window: np.ndarray, method: str) -> np.ndarray:
    """Single-window covariance estimate. Returns SPD matrix."""
    window = window - window.mean(axis=1, keepdims=True)

    if method == "empirical":
        n = window.shape[1]
        return (window @ window.T) / n

    from sklearn.covariance import OAS, LedoitWolf
    estimator = OAS() if method == "oas" else LedoitWolf()
    estimator.fit(window.T)  # sklearn expects (n_samples, n_features)
    return estimator.covariance_


def is_spd(matrix: np.ndarray, tol: float = 1e-12) -> bool:
    """Check that a matrix is symmetric positive-definite."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.allclose(matrix, matrix.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(matrix)
    return bool(np.all(eigvals > tol))
