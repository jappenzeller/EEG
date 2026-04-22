"""First-moment projection (polarity probe) between two labeled SPD sets.

Computes the signed projection of the difference of class-conditional
Frechet means onto a reference direction. Sign is the polarity; magnitude
is the shift.

This module implements the *instrument reading*, not the framework.
The reference direction is caller-supplied so no QNFM sign convention
is baked in.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, logm
from scipy.linalg import fractional_matrix_power


def frechet_mean_bw(
    covs: np.ndarray, max_iter: int = 50, tol: float = 1e-8,
) -> np.ndarray:
    """Bures-Wasserstein Frechet mean of SPD matrices.

    Iterative fixed-point starting from the arithmetic mean.
    BW is the metric used in the QNFM empirical spine.
    """
    n = covs.shape[0]
    mean = covs.mean(axis=0)
    for _ in range(max_iter):
        sqrt_mean = fractional_matrix_power(mean, 0.5)
        summed = np.zeros_like(mean)
        for c in covs:
            middle = sqrt_mean @ c @ sqrt_mean
            summed += fractional_matrix_power(middle, 0.5)
        new_mean = summed / n
        if np.linalg.norm(new_mean - mean, "fro") < tol:
            mean = new_mean
            break
        mean = new_mean
    return 0.5 * (mean + mean.T)


def tangent_log_map(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Log map at `base` of `point` in the BW tangent space.

    Returns the tangent vector (symmetric matrix):
        B^{1/2} log(B^{-1/2} P B^{-1/2}) B^{1/2}
    """
    sqrt_base = fractional_matrix_power(base, 0.5)
    inv_sqrt_base = fractional_matrix_power(base, -0.5)
    whitened = inv_sqrt_base @ point @ inv_sqrt_base
    log_whitened = logm(whitened)
    return sqrt_base @ log_whitened @ sqrt_base


def polarity_projection(
    state_a_covs: np.ndarray,
    state_b_covs: np.ndarray,
    reference_direction: np.ndarray,
) -> dict:
    """Project the (state_b - state_a) tangent shift onto a reference.

    Args:
        state_a_covs: (n_a, C, C) covariance matrices for state A
        state_b_covs: (n_b, C, C) covariance matrices for state B
        reference_direction: (C, C) symmetric tangent vector

    Returns:
        dict with 'sign', 'magnitude', 'projection',
        'population_mean', 'mean_a', 'mean_b'
    """
    combined = np.concatenate([state_a_covs, state_b_covs], axis=0)
    pop_mean = frechet_mean_bw(combined)
    mean_a = frechet_mean_bw(state_a_covs)
    mean_b = frechet_mean_bw(state_b_covs)

    shift = (
        tangent_log_map(pop_mean, mean_b)
        - tangent_log_map(pop_mean, mean_a)
    )
    ref_norm = np.linalg.norm(reference_direction, "fro")
    if ref_norm < 1e-12:
        raise ValueError("reference_direction has near-zero norm")

    projection = np.sum(shift * reference_direction) / ref_norm
    return {
        "sign": int(np.sign(projection)) if projection != 0 else 0,
        "magnitude": float(abs(projection)),
        "projection": float(projection),
        "population_mean": pop_mean,
        "mean_a": mean_a,
        "mean_b": mean_b,
    }


def reference_from_class_difference(
    state_a_covs: np.ndarray,
    state_b_covs: np.ndarray,
) -> np.ndarray:
    """Build a reference direction from the top eigenvector of the
    log-Euclidean mean difference.

    Self-calibrated: sign is arbitrary (eigenvector convention). Useful
    for magnitude but NOT for cross-session sign comparison. For that,
    fix a reference from a calibration session and reuse it.
    """
    mean_a = frechet_mean_bw(state_a_covs)
    mean_b = frechet_mean_bw(state_b_covs)
    diff = logm(mean_b) - logm(mean_a)
    diff_sym = 0.5 * (diff + diff.T)
    eigvals, eigvecs = eigh(diff_sym)
    top = eigvecs[:, np.argmax(np.abs(eigvals))]
    return np.outer(top, top)
