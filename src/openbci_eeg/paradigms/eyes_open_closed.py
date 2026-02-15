"""
Eyes Open/Closed paradigm for alpha rhythm validation.

This is the gating test for EEG quality: if we can't detect the alpha power
increase over O1/O2/Pz during eyes closed, nothing downstream will work.

Protocol:
    5 min eyes open → 5 min eyes closed → repeat 3x

Expected signals:
    - Alpha (8-13 Hz) power increase over occipital/parietal during eyes closed
    - Alpha suppression (desynchronization) during eyes open

This paradigm is:
    - The simplest, most reproducible EEG state transition
    - Essential hardware/signal quality validation
    - Baseline for all other paradigms
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EyesOpenClosedConfig:
    """Configuration for eyes open/closed paradigm."""
    eyes_open_sec: float = 300.0  # 5 minutes
    eyes_closed_sec: float = 300.0  # 5 minutes
    n_cycles: int = 3
    countdown_sec: float = 5.0  # Countdown before condition change
    beep_on_transition: bool = True


class EyesOpenClosedParadigm:
    """
    Eyes Open/Closed paradigm controller.

    Manages timing, cues, and segment labeling for the protocol.
    """

    CONDITION_EYES_OPEN = 0
    CONDITION_EYES_CLOSED = 1

    def __init__(self, config: Optional[EyesOpenClosedConfig] = None):
        self.config = config or EyesOpenClosedConfig()
        self.events: list[dict] = []
        self._start_time: Optional[float] = None
        self._current_condition: Optional[int] = None

    @property
    def total_duration_sec(self) -> float:
        """Total paradigm duration in seconds."""
        cycle_duration = self.config.eyes_open_sec + self.config.eyes_closed_sec
        return cycle_duration * self.config.n_cycles

    def get_schedule(self) -> list[dict]:
        """
        Generate the paradigm schedule.

        Returns:
            List of dicts with 'start_sec', 'duration_sec', 'condition', 'label'.
        """
        schedule = []
        t = 0.0

        for cycle in range(self.config.n_cycles):
            # Eyes open
            schedule.append({
                "start_sec": t,
                "duration_sec": self.config.eyes_open_sec,
                "condition": self.CONDITION_EYES_OPEN,
                "label": "eyes_open",
                "cycle": cycle + 1,
            })
            t += self.config.eyes_open_sec

            # Eyes closed
            schedule.append({
                "start_sec": t,
                "duration_sec": self.config.eyes_closed_sec,
                "condition": self.CONDITION_EYES_CLOSED,
                "label": "eyes_closed",
                "cycle": cycle + 1,
            })
            t += self.config.eyes_closed_sec

        return schedule

    def run_interactive(self, callback=None) -> None:
        """
        Run paradigm interactively with console prompts.

        Args:
            callback: Optional function called at each transition with
                (condition, elapsed_sec, remaining_sec).
        """
        schedule = self.get_schedule()
        self._start_time = time.time()

        print("\n" + "=" * 50)
        print("EYES OPEN/CLOSED PARADIGM")
        print("=" * 50)
        print(f"Total duration: {self.total_duration_sec / 60:.1f} minutes")
        print(f"Cycles: {self.config.n_cycles}")
        print("=" * 50 + "\n")

        for i, block in enumerate(schedule):
            condition = block["condition"]
            duration = block["duration_sec"]
            label = block["label"].upper().replace("_", " ")

            # Countdown
            print(f"\nStarting in {self.config.countdown_sec:.0f} seconds...")
            for sec in range(int(self.config.countdown_sec), 0, -1):
                print(f"  {sec}...")
                time.sleep(1)

            # Condition cue
            if self.config.beep_on_transition:
                _beep()

            print("\n" + "-" * 40)
            print(f">>> {label} <<<")
            print(f"Duration: {duration / 60:.1f} minutes")
            print(f"Cycle {block['cycle']} of {self.config.n_cycles}")
            print("-" * 40)

            self._current_condition = condition
            self.events.append({
                "timestamp": time.time() - self._start_time,
                "event": "condition_start",
                "condition": condition,
                "label": block["label"],
            })

            if callback:
                callback(condition, 0, duration)

            # Wait for duration with progress updates
            start = time.time()
            while time.time() - start < duration:
                elapsed = time.time() - start
                remaining = duration - elapsed

                # Progress update every 30 seconds
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    mins_remaining = remaining / 60
                    print(f"  {mins_remaining:.1f} min remaining...")

                if callback:
                    callback(condition, elapsed, remaining)

                time.sleep(1)

            self.events.append({
                "timestamp": time.time() - self._start_time,
                "event": "condition_end",
                "condition": condition,
            })

        print("\n" + "=" * 50)
        print("PARADIGM COMPLETE")
        print("=" * 50)

        if self.config.beep_on_transition:
            _beep()
            time.sleep(0.2)
            _beep()

    def label_segments(
        self,
        timestamps: np.ndarray,
        segment_duration_sec: float = 30.0,
    ) -> np.ndarray:
        """
        Generate condition labels for EEG segments.

        Args:
            timestamps: 1D array of segment start times (in seconds from paradigm start).
            segment_duration_sec: Duration of each segment.

        Returns:
            1D array of condition labels (0=eyes_open, 1=eyes_closed, -1=transition).
        """
        schedule = self.get_schedule()
        labels = np.full(len(timestamps), -1, dtype=int)

        for i, t in enumerate(timestamps):
            # Find which block this segment falls into
            for block in schedule:
                block_start = block["start_sec"]
                block_end = block_start + block["duration_sec"]

                # Check if segment is fully within this block
                segment_end = t + segment_duration_sec
                if t >= block_start and segment_end <= block_end:
                    labels[i] = block["condition"]
                    break

        return labels

    def save_events(self, path: Path) -> None:
        """Save recorded events to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump({
                "config": {
                    "eyes_open_sec": self.config.eyes_open_sec,
                    "eyes_closed_sec": self.config.eyes_closed_sec,
                    "n_cycles": self.config.n_cycles,
                },
                "events": self.events,
            }, f, indent=2)

        logger.info("Events saved to %s", path)


def _beep():
    """Cross-platform beep sound."""
    try:
        import sys
        if sys.platform == "win32":
            import winsound
            winsound.Beep(800, 200)
        else:
            print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)


def validate_alpha_response(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    sfreq: float = 125.0,
    channel_names: Optional[list[str]] = None,
    alpha_channels: Optional[list[str]] = None,
) -> dict:
    """
    Validate that eyes-closed condition shows increased alpha power.

    This is the gating test: if alpha doesn't increase with eyes closed,
    there's a signal quality problem.

    Args:
        eeg_data: 2D array of shape (n_channels, n_samples).
        labels: Segment labels (0=eyes_open, 1=eyes_closed).
        sfreq: Sample rate.
        channel_names: Names for each channel.
        alpha_channels: Channels to check for alpha (default: O1, O2, Pz).

    Returns:
        Dict with validation results and alpha power statistics.
    """
    from openbci_eeg.pn_extraction.band_power import compute_band_powers, BandConfig

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(eeg_data.shape[0])]

    if alpha_channels is None:
        # Default to occipital/parietal
        alpha_channels = ["O1", "O2", "Pz"]
        alpha_channels = [ch for ch in alpha_channels if ch in channel_names]
        if not alpha_channels:
            alpha_channels = channel_names[:3]  # Fallback to first 3

    # Find channel indices
    ch_indices = [channel_names.index(ch) for ch in alpha_channels]

    # Segment data by condition
    eyes_open_mask = labels == 0
    eyes_closed_mask = labels == 1

    results = {
        "channels_tested": alpha_channels,
        "alpha_power_eyes_open": {},
        "alpha_power_eyes_closed": {},
        "alpha_increase_ratio": {},
        "validation_passed": False,
    }

    bands = BandConfig()
    all_ratios = []

    for ch_name, ch_idx in zip(alpha_channels, ch_indices):
        # Compute mean alpha power for each condition
        # (In practice, would segment the data properly)

        # Simplified: use full data with mask
        alpha_open = []
        alpha_closed = []

        # Estimate alpha power (would need actual segmentation in production)
        n_samples = eeg_data.shape[1]
        n_segments = len(labels)
        samples_per_segment = n_samples // n_segments

        for seg_idx, label in enumerate(labels):
            start = seg_idx * samples_per_segment
            end = start + samples_per_segment
            if end > n_samples:
                break

            segment = eeg_data[ch_idx, start:end]
            bp = compute_band_powers(segment, sfreq, bands=bands)

            if label == 0:
                alpha_open.append(bp["alpha"])
            elif label == 1:
                alpha_closed.append(bp["alpha"])

        mean_open = np.mean(alpha_open) if alpha_open else 0
        mean_closed = np.mean(alpha_closed) if alpha_closed else 0
        ratio = mean_closed / (mean_open + 1e-10)

        results["alpha_power_eyes_open"][ch_name] = float(mean_open)
        results["alpha_power_eyes_closed"][ch_name] = float(mean_closed)
        results["alpha_increase_ratio"][ch_name] = float(ratio)

        all_ratios.append(ratio)

    # Validation: alpha should increase by at least 20% with eyes closed
    mean_ratio = np.mean(all_ratios)
    results["mean_alpha_ratio"] = float(mean_ratio)
    results["validation_passed"] = mean_ratio > 1.2  # 20% increase

    if results["validation_passed"]:
        logger.info("Alpha validation PASSED: %.1f%% increase with eyes closed",
                    (mean_ratio - 1) * 100)
    else:
        logger.warning("Alpha validation FAILED: only %.1f%% change with eyes closed",
                       (mean_ratio - 1) * 100)

    return results
