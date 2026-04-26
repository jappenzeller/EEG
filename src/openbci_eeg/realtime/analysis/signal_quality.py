"""Signal quality metrics -- streaming, cheap, called per UI tick.

Three complementary checks + suspiciously-quiet rule:
    1. Railed percentage: samples exceeding the ADC saturation threshold.
    2. Standard deviation: catches flatlines (dead channels) and excessive noise.
    3. Line-noise ratio: fraction of variance at 60 Hz (floating electrode proxy).
    4. Suspicious quiet: bias-pinned channels with no real signal (low 60Hz + moderate std).

Per-channel status is the worst-of all checks.

Two threshold profiles:
    BENCH — SRB-only bench testing, no BIAS. Tuned 2026-04-19.
    SCALP — Full prep with SRB + BIAS + gel electrodes on scalp.

Calibration notes (update as real-world testing produces data):
    - Floating gold cup in air: std ~400-2000 uV, 60 Hz ratio >85%
    - Gel finger, SRB only: std ~70-210 uV, 60Hz 30-85% -> YELLOW
    - Gel finger, best contact (F7): std 48.8 uV, 60Hz 0% -> GREEN (1 tick)
    - Bias-pinned (no contact, Daisy): std 50-170 uV, 60Hz 0% -> false YELLOW
    - BIAS feedback latch (low impedance): std 0.0, flatline -> RED
    - Full prep target (SRB + BIAS + gel scalp): std ~5-20 uV -> GREEN
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import iirnotch, sosfilt, tf2sos

# Status codes
STATUS_GREEN = 0
STATUS_YELLOW = 1
STATUS_RED = 2

# Line frequency (module-level, used by line_noise_ratio)
LINE_FREQ_HZ: float = 60.0  # 50.0 if outside North America


# ---------------------------------------------------------------------------
# Threshold profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """Signal quality thresholds. Two defaults: BENCH and SCALP."""
    name: str

    # Rail check (% of samples with |deviation from mean| > threshold)
    rail_threshold_uv: float
    rail_green_max_pct: float
    rail_yellow_max_pct: float

    # Std check (per-channel standard deviation in uV)
    flatline_std_uv: float
    low_activity_std_uv: float
    high_activity_std_uv: float
    saturation_std_uv: float

    # Line noise ratio (fraction of variance at 60 Hz)
    line_noise_green_max: float
    line_noise_yellow_max: float

    # Suspiciously-quiet: bias-pinned channels with no real signal.
    # Trigger when std > min AND line_ratio < max AND rail% < 1.
    # Forces at-least-yellow regardless of other rules.
    suspicious_quiet_std_min: float
    suspicious_quiet_line_max: float

    @property
    def description(self) -> str:
        if self.name == "bench":
            return "SRB-only / dry contact. GREEN is optimistic."
        elif self.name == "scalp":
            return "expects BIAS connected and paste electrodes"
        return self.name


BENCH = Profile(
    name="bench",
    rail_threshold_uv=500.0,
    rail_green_max_pct=5.0,
    rail_yellow_max_pct=50.0,
    flatline_std_uv=1.0,
    low_activity_std_uv=5.0,
    high_activity_std_uv=50.0,
    saturation_std_uv=500.0,
    line_noise_green_max=0.30,
    line_noise_yellow_max=0.85,
    suspicious_quiet_std_min=50.0,
    suspicious_quiet_line_max=0.05,
)

SCALP = Profile(
    name="scalp",
    rail_threshold_uv=200.0,
    rail_green_max_pct=2.0,
    rail_yellow_max_pct=20.0,
    flatline_std_uv=1.0,
    low_activity_std_uv=3.0,
    high_activity_std_uv=25.0,
    saturation_std_uv=100.0,
    line_noise_green_max=0.15,
    line_noise_yellow_max=0.50,
    suspicious_quiet_std_min=30.0,
    suspicious_quiet_line_max=0.05,
)

PROFILES: dict[str, Profile] = {"bench": BENCH, "scalp": SCALP}


# ---------------------------------------------------------------------------
# Wire colors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canonical channel mapping (single source of truth: CANONICAL_WIRING.md)
# ---------------------------------------------------------------------------

CHANNEL_POSITIONS: dict[int, str] = {
    1: "Fp1",  2: "Fp2",  3: "F3",   4: "F4",
    5: "F7",   6: "F8",   7: "Fz",   8: "Oz",
    9: "C3",  10: "C4",  11: "Pz",  12: "Cz",
    13: "P3", 14: "O2",  15: "O1",  16: "P4",
}

WIRE_COLORS: dict[int, str] = {
    1:  "#9AA4AE",  # Fp1  (Cyton ch 1, grey)
    2:  "#8A2BE2",  # Fp2  (Cyton ch 2, purple)
    3:  "#1E90FF",  # F3   (Cyton ch 3, blue)
    4:  "#2E8B57",  # F4   (Cyton ch 4, green)
    5:  "#FFD700",  # F7   (Cyton ch 5, yellow)
    6:  "#FF8C00",  # F8   (Cyton ch 6, orange)
    7:  "#D9342B",  # Fz   (Cyton ch 7, red)
    8:  "#8B5A2B",  # Oz   (Cyton ch 8, brown)
    9:  "#9AA4AE",  # C3   (Daisy ch 9, grey)
    10: "#8A2BE2",  # C4   (Daisy ch 10, purple)
    11: "#1E90FF",  # Pz   (Daisy ch 11, blue)
    12: "#2E8B57",  # Cz   (Daisy ch 12, green)
    13: "#FFD700",  # P3   (Daisy ch 13, yellow)
    14: "#FF8C00",  # O2   (Daisy ch 14, orange)
    15: "#D9342B",  # O1   (Daisy ch 15, red)
    16: "#8B5A2B",  # P4   (Daisy ch 16, brown)
}


def channel_board(channel_number: int) -> str:
    """Return 'cyton' (ch 1-8) or 'daisy' (ch 9-16)."""
    if 1 <= channel_number <= 8:
        return "cyton"
    if 9 <= channel_number <= 16:
        return "daisy"
    raise ValueError(f"channel_number {channel_number} out of range 1..16")


def channel_position(channel_number: int) -> str:
    """Channel number (1-indexed) to 10-20 position name."""
    return CHANNEL_POSITIONS[channel_number]


def wire_color(channel_number: int) -> str:
    """Channel number (1-indexed, 1-16) to hex wire color."""
    return WIRE_COLORS[channel_number]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_notch_cache: dict[tuple[float, float], np.ndarray] = {}


def _notch_sos(fs: float, freq: float = LINE_FREQ_HZ, q: float = 30.0) -> np.ndarray:
    """Cache SOS coefficients for a narrow notch. Same filter every call."""
    key = (fs, freq)
    if key not in _notch_cache:
        b, a = iirnotch(freq, q, fs)
        _notch_cache[key] = tf2sos(b, a)
    return _notch_cache[key]


def railed_percent(data: np.ndarray, threshold_uv: float = 500.0) -> np.ndarray:
    """Per-channel percent of samples with |deviation from mean| > threshold.

    Subtracts the per-channel mean before checking, so DC offsets from the
    SRB reference path don't cause false railing.
    """
    if data.size == 0:
        return np.zeros(data.shape[0], dtype=np.float32)
    centered = data - data.mean(axis=1, keepdims=True)
    return (np.abs(centered) > threshold_uv).mean(axis=1).astype(np.float32) * 100.0


def channel_std(data: np.ndarray) -> np.ndarray:
    """Per-channel standard deviation in uV. Shape (n_channels,)."""
    if data.size == 0:
        return np.zeros(data.shape[0], dtype=np.float32)
    return data.std(axis=1).astype(np.float32)


def line_noise_ratio(data: np.ndarray, fs: float) -> np.ndarray:
    """Fraction of variance at the line frequency. Shape (n_channels,), range [0, 1]."""
    if data.size == 0:
        return np.zeros(data.shape[0], dtype=np.float32)

    raw_std = data.std(axis=1)
    notched = sosfilt(_notch_sos(fs), data, axis=1)
    notched_std = notched.std(axis=1)

    eps = 1e-9
    ratio = 1.0 - (notched_std / (raw_std + eps)) ** 2
    return np.clip(ratio, 0.0, 1.0).astype(np.float32)


def channel_status(
    railed_pct: np.ndarray,
    std_uv: np.ndarray,
    line_ratio: np.ndarray,
    profile: Profile = BENCH,
) -> np.ndarray:
    """Return a (n_channels,) int array: 0=green, 1=yellow, 2=red.

    Takes the worst of rail, std, line-noise, and suspicious-quiet checks.
    """
    n = len(railed_pct)
    status = np.zeros(n, dtype=np.int8)

    # Suspiciously-quiet: bias-pinned channel with no real signal.
    sus = (
        (std_uv > profile.suspicious_quiet_std_min)
        & (line_ratio < profile.suspicious_quiet_line_max)
        & (railed_pct < 1.0)
    )

    # yellow conditions
    yellow = (
        sus
        | (railed_pct >= profile.rail_green_max_pct)
        | (std_uv < profile.low_activity_std_uv)
        | (std_uv > profile.high_activity_std_uv)
        | (line_ratio > profile.line_noise_green_max)
    )
    status[yellow] = STATUS_YELLOW

    # red conditions (override yellow)
    red = (
        (railed_pct >= profile.rail_yellow_max_pct)
        | (std_uv < profile.flatline_std_uv)
        | (std_uv > profile.saturation_std_uv)
        | (line_ratio > profile.line_noise_yellow_max)
    )
    status[red] = STATUS_RED

    return status
