"""Signal quality metrics -- railed percentage, channel std, line-noise ratio,
profiles, suspicious-quiet rule, and composite status."""

import numpy as np
import pytest

from openbci_eeg.realtime.analysis.signal_quality import (
    railed_percent,
    channel_std,
    channel_status,
    line_noise_ratio,
    wire_color,
    channel_position,
    channel_board,
    WIRE_COLORS,
    CHANNEL_POSITIONS,
    Profile,
    BENCH,
    SCALP,
    PROFILES,
)

FS = 125.0


def _sine(freq, duration_s, amplitude, fs=FS):
    t = np.arange(int(duration_s * fs)) / fs
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- railed_percent tests ---------------------------------------------------

def test_empty_input_returns_zeros():
    data = np.zeros((16, 0), dtype=np.float32)
    out = railed_percent(data)
    assert out.shape == (16,)
    assert np.all(out == 0.0)


def test_no_railed_samples():
    data = np.ones((4, 100), dtype=np.float32) * 50.0
    out = railed_percent(data, threshold_uv=150.0)
    assert np.all(out == 0.0)


def test_all_railed():
    data = np.tile(np.array([200.0, -200.0], dtype=np.float32), (4, 50))
    out = railed_percent(data, threshold_uv=150.0)
    assert np.all(out == 100.0)


def test_partial_railed_exact():
    data = np.zeros((1, 100), dtype=np.float32)
    data[0, :30] = 500.0
    out = railed_percent(data, threshold_uv=150.0)
    np.testing.assert_allclose(out[0], 30.0, atol=0.01)


def test_threshold_exact_boundary_not_railed():
    data = np.ones((1, 10), dtype=np.float32) * 150.0
    out = railed_percent(data, threshold_uv=150.0)
    assert out[0] == 0.0


def test_negative_values_also_count():
    data = np.tile(np.array([0.0, -400.0], dtype=np.float32), (1, 5))
    out = railed_percent(data, threshold_uv=150.0)
    assert out[0] == 100.0


def test_dc_offset_does_not_cause_false_railing():
    rng = np.random.default_rng(42)
    data = rng.normal(5000.0, 10.0, size=(4, 250)).astype(np.float32)
    out = railed_percent(data, threshold_uv=150.0)
    assert np.all(out == 0.0)


# --- channel_std tests -------------------------------------------------------

def test_channel_std_empty():
    assert channel_std(np.zeros((16, 0), dtype=np.float32)).shape == (16,)


def test_channel_std_known_value():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 10.0, size=(4, 1000)).astype(np.float32)
    out = channel_std(data)
    np.testing.assert_allclose(out, 10.0, rtol=0.1)


# --- line_noise_ratio tests --------------------------------------------------

def test_line_noise_ratio_pure_60hz():
    sig = _sine(60.0, 2.0, 20.0).reshape(1, -1)
    r = line_noise_ratio(sig, FS)
    assert r[0] > 0.9


def test_line_noise_ratio_pure_10hz():
    sig = _sine(10.0, 2.0, 20.0).reshape(1, -1)
    r = line_noise_ratio(sig, FS)
    assert r[0] < 0.1


def test_line_noise_ratio_mixed():
    sig = (_sine(10.0, 2.0, 20.0) + _sine(60.0, 2.0, 20.0)).reshape(1, -1)
    r = line_noise_ratio(sig, FS)
    assert 0.3 < r[0] < 0.7


def test_line_noise_ratio_flatline_returns_zero_not_nan():
    sig = np.ones((1, 250), dtype=np.float32) * 5.0
    r = line_noise_ratio(sig, FS)
    assert r[0] == 0.0


def test_line_noise_ratio_empty():
    r = line_noise_ratio(np.zeros((4, 0), dtype=np.float32), FS)
    assert r.shape == (4,)
    assert np.all(r == 0.0)


# --- channel_status tests (with BENCH profile) --------------------------------

def test_status_flatline_is_red():
    data = np.ones((1, 500), dtype=np.float32) * 20.0
    std = channel_std(data)
    assert std[0] < BENCH.flatline_std_uv
    status = channel_status(np.array([0.0]), std, np.array([0.0]), BENCH)
    assert status[0] == 2


def test_status_normal_eeg_is_green():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 15.0, size=(1, 500)).astype(np.float32)
    std = channel_std(data)
    status = channel_status(np.array([0.0]), std, np.array([0.15]), BENCH)
    assert status[0] == 0


def test_status_low_activity_is_yellow():
    rng = np.random.default_rng(2)
    data = rng.normal(0, 2.0, size=(1, 500)).astype(np.float32)
    std = channel_std(data)
    assert BENCH.flatline_std_uv < std[0] < BENCH.low_activity_std_uv
    assert channel_status(np.array([0.0]), std, np.array([0.0]), BENCH)[0] == 1


def test_status_worst_of_multiple_problems():
    status = channel_status(
        np.array([60.0]), np.array([15.0], dtype=np.float32),
        np.array([0.0]), BENCH)
    assert status[0] == 2


def test_status_yellow_rail_beats_green_std():
    status = channel_status(
        np.array([10.0]), np.array([15.0], dtype=np.float32),
        np.array([0.0]), BENCH)
    assert status[0] == 1


def test_status_high_noise_is_yellow():
    rng = np.random.default_rng(3)
    data = rng.normal(0, 90.0, size=(1, 500)).astype(np.float32)
    std = channel_std(data)
    assert BENCH.high_activity_std_uv < std[0] < BENCH.saturation_std_uv
    assert channel_status(np.array([0.0]), std, np.array([0.0]), BENCH)[0] == 1


def test_status_line_noise_dominates():
    status = channel_status(
        railed_pct=np.array([0.0]),
        std_uv=np.array([15.0], dtype=np.float32),
        line_ratio=np.array([0.95]),
        profile=BENCH,
    )
    assert status[0] == 2


def test_status_clean_eeg_is_green():
    status = channel_status(
        railed_pct=np.array([0.0]),
        std_uv=np.array([15.0], dtype=np.float32),
        line_ratio=np.array([0.15]),
        profile=BENCH,
    )
    assert status[0] == 0


# --- profile tests -----------------------------------------------------------

def test_profiles_loadable():
    assert BENCH.name == "bench"
    assert SCALP.name == "scalp"
    assert SCALP.high_activity_std_uv < BENCH.high_activity_std_uv


def test_profiles_dict():
    assert set(PROFILES.keys()) == {"bench", "scalp"}
    assert PROFILES["bench"] is BENCH
    assert PROFILES["scalp"] is SCALP


def test_suspicious_quiet_forces_yellow_on_bench():
    # Bias-pinned: std=60 (above sus_min=50), line=0.01 (<0.05), rail=0
    status = channel_status(
        railed_pct=np.array([0.0]),
        std_uv=np.array([60.0], dtype=np.float32),
        line_ratio=np.array([0.01]),
        profile=BENCH,
    )
    assert status[0] == 1  # yellow from suspicious quiet


def test_suspicious_quiet_does_not_trigger_with_line_noise():
    # Same std but 60Hz=20% -> not suspiciously quiet, it's a real signal
    status = channel_status(
        railed_pct=np.array([0.0]),
        std_uv=np.array([40.0], dtype=np.float32),
        line_ratio=np.array([0.20]),
        profile=BENCH,
    )
    assert status[0] == 0  # green


def test_scalp_profile_rejects_bench_good():
    # std=40 is green on bench but yellow on scalp (>25 ceiling)
    status_bench = channel_status(
        np.array([0.0]), np.array([40.0], dtype=np.float32),
        np.array([0.1]), BENCH)
    status_scalp = channel_status(
        np.array([0.0]), np.array([40.0], dtype=np.float32),
        np.array([0.1]), SCALP)
    assert status_bench[0] == 0
    assert status_scalp[0] >= 1


def test_scalp_profile_accepts_realistic_scalp():
    # Real scalp-with-BIAS: std=12, 60Hz=10%, rail=0 -> green on SCALP
    status = channel_status(
        np.array([0.0]), np.array([12.0], dtype=np.float32),
        np.array([0.10]), SCALP)
    assert status[0] == 0


# --- wire_color tests --------------------------------------------------------

def test_wire_color_pairs_match():
    for ch in range(1, 9):
        assert wire_color(ch) == wire_color(ch + 8)


def test_wire_color_full_coverage():
    assert set(WIRE_COLORS.keys()) == set(range(1, 17))


def test_wire_color_reverse_rainbow():
    assert wire_color(8).lower() == "#8b5a2b"
    assert wire_color(1).lower() == "#9aa4ae"


def test_wire_color_invalid_channel_raises():
    with pytest.raises(KeyError):
        wire_color(17)
