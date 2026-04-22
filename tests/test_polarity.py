"""Polarity probe tests -- verify geometric properties, not research outcomes."""

import numpy as np
import pytest

from openbci_eeg.realtime.analysis.polarity import (
    frechet_mean_bw,
    polarity_projection,
    reference_from_class_difference,
)


def _random_spd(n, rng, scale=1.0):
    A = rng.normal(0, scale, size=(n, n))
    return A @ A.T + np.eye(n) * 0.1


def test_frechet_mean_is_spd():
    rng = np.random.default_rng(0)
    covs = np.array([_random_spd(8, rng) for _ in range(20)])
    mean = frechet_mean_bw(covs, max_iter=20)
    assert np.allclose(mean, mean.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(mean) > 0)


def test_polarity_sign_flips_when_states_swap():
    rng = np.random.default_rng(1)
    state_a = np.array([_random_spd(6, rng, scale=1.0) for _ in range(15)])
    state_b = np.array([_random_spd(6, rng, scale=1.5) for _ in range(15)])
    ref = reference_from_class_difference(state_a, state_b)
    ab = polarity_projection(state_a, state_b, ref)
    ba = polarity_projection(state_b, state_a, ref)
    assert ab["sign"] == -ba["sign"]
    assert ab["magnitude"] == pytest.approx(ba["magnitude"], rel=1e-6)


def test_polarity_magnitude_small_when_states_similar():
    rng = np.random.default_rng(2)
    covs = np.array([_random_spd(6, rng) for _ in range(20)])
    a = covs[:10]
    b = covs[10:]
    ref = reference_from_class_difference(a, b)
    result = polarity_projection(a, b, ref)
    # Same distribution, so magnitude should be much smaller than when
    # states differ in scale (test_polarity_sign_flips uses scale 1.0 vs 1.5)
    rng2 = np.random.default_rng(1)
    diff_a = np.array([_random_spd(6, rng2, scale=1.0) for _ in range(15)])
    diff_b = np.array([_random_spd(6, rng2, scale=1.5) for _ in range(15)])
    ref2 = reference_from_class_difference(diff_a, diff_b)
    result_diff = polarity_projection(diff_a, diff_b, ref2)
    assert result["magnitude"] < result_diff["magnitude"]


def test_reference_direction_is_symmetric():
    rng = np.random.default_rng(3)
    a = np.array([_random_spd(6, rng) for _ in range(10)])
    b = np.array([_random_spd(6, rng, scale=1.5) for _ in range(10)])
    ref = reference_from_class_difference(a, b)
    assert np.allclose(ref, ref.T, atol=1e-8)
