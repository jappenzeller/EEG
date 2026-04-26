"""Enforce cross-file agreement on the wiring mapping."""

from pathlib import Path

from openbci_eeg.realtime.analysis.signal_quality import (
    CHANNEL_POSITIONS,
    WIRE_COLORS,
    channel_board,
    channel_position,
    wire_color,
)
from openbci_eeg import CHANNEL_NAMES

REPO = Path(__file__).resolve().parent.parent


def test_positions_full_coverage():
    assert set(CHANNEL_POSITIONS.keys()) == set(range(1, 17))


def test_wire_colors_full_coverage():
    assert set(WIRE_COLORS.keys()) == set(range(1, 17))


def test_cyton_daisy_share_wire_colors_per_column():
    for ch in range(1, 9):
        assert WIRE_COLORS[ch] == WIRE_COLORS[ch + 8], (
            f"ch {ch} and ch {ch + 8} must share wire color"
        )


def test_cyton_and_daisy_assignment():
    for ch in range(1, 9):
        assert channel_board(ch) == "cyton"
    for ch in range(9, 17):
        assert channel_board(ch) == "daisy"


def test_known_positions():
    """Spot check the ones that have been mixed up historically."""
    assert channel_position(7) == "Fz"    # was T7, then Fz
    assert channel_position(8) == "Oz"    # was T8, then Oz
    assert channel_position(11) == "Pz"   # verified 2026-04-19
    assert channel_position(12) == "Cz"   # verified 2026-04-19
    assert channel_position(14) == "O2"   # was P4, swapped 2026-04-26
    assert channel_position(15) == "O1"
    assert channel_position(16) == "P4"   # was O2, swapped 2026-04-26


def test_full_midline_spine_present():
    positions = set(CHANNEL_POSITIONS.values())
    for p in ["Fz", "Cz", "Pz", "Oz"]:
        assert p in positions, f"{p} should be in the current montage"


def test_bilateral_temporal_removed():
    positions = set(CHANNEL_POSITIONS.values())
    assert "T7" not in positions
    assert "T8" not in positions


def test_p7_p8_not_in_montage():
    positions = set(CHANNEL_POSITIONS.values())
    assert "P7" not in positions
    assert "P8" not in positions


def test_known_wire_colors():
    """Spot check: grey on ch 1/9, brown on ch 8/16 (rainbow reverse)."""
    assert wire_color(1).upper() == "#9AA4AE"
    assert wire_color(9).upper() == "#9AA4AE"
    assert wire_color(8).upper() == "#8B5A2B"
    assert wire_color(16).upper() == "#8B5A2B"


def test_canonical_wiring_md_exists():
    assert (REPO / "CANONICAL_WIRING.md").exists(), (
        "CANONICAL_WIRING.md must exist at repo root as the single source of truth"
    )


def test_o2_not_on_slow_channel():
    """O2 must be on Daisy ch 14 (faster), not ch 16 (slowest)."""
    assert CHANNEL_POSITIONS[14] == "O2"
    assert CHANNEL_POSITIONS[16] != "O2"


def test_bilateral_occipital_present():
    """Both O1 and O2 must be in the montage for bilateral alpha checks."""
    positions = set(CHANNEL_POSITIONS.values())
    assert "O1" in positions
    assert "O2" in positions


def test_channel_names_matches_positions():
    """Package-level CHANNEL_NAMES must agree with CHANNEL_POSITIONS."""
    for i, name in enumerate(CHANNEL_NAMES):
        ch = i + 1
        assert name == CHANNEL_POSITIONS[ch], (
            f"CHANNEL_NAMES[{i}]={name} but CHANNEL_POSITIONS[{ch}]={CHANNEL_POSITIONS[ch]}"
        )
