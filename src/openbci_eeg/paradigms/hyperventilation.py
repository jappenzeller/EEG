"""
Hyperventilation (HV) paradigm for seizure-like EEG activation.

Standard clinical EEG activation procedure that produces the most seizure-like
EEG changes in healthy subjects: generalized slowing (increased delta/theta).

Protocol:
    3 min normal breathing → 3 min hyperventilation → 5 min recovery → repeat 1x

Expected signals:
    - Generalized slowing during HV (increased delta/theta power)
    - Possible spike-wave if predisposed (rare in healthy subjects)
    - Return to baseline during recovery

SAFETY WARNINGS:
    - Mild dizziness/lightheadedness is NORMAL and expected
    - Subject must be SEATED with head support (syncope risk)
    - Partner/observer REQUIRED (not solo recording)
    - Pulse oximeter optional but recommended
    - ABORT if: sustained dizziness >30s, SpO2 <92%, numbness/tingling persists,
      visual disturbances, chest pain, or any concerning symptoms
    - Keep water nearby
    - Subject can stop at any time

This paradigm produces E-I shifts closest to actual seizure dynamics
without being pathological.
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
class HyperventilationConfig:
    """Configuration for hyperventilation paradigm."""
    baseline_sec: float = 180.0  # 3 minutes normal breathing
    hyperventilation_sec: float = 180.0  # 3 minutes HV
    recovery_sec: float = 300.0  # 5 minutes recovery
    n_cycles: int = 1  # Usually just 1 cycle clinically
    breathing_rate_per_min: float = 25.0  # Target breaths/min during HV
    countdown_sec: float = 10.0  # Longer countdown for safety
    require_safety_acknowledgment: bool = True


class HyperventilationParadigm:
    """
    Hyperventilation paradigm controller.

    Manages timing, safety checks, breathing cues, and segment labeling.
    """

    CONDITION_BASELINE = 0
    CONDITION_HYPERVENTILATION = 1
    CONDITION_RECOVERY = 2

    def __init__(self, config: Optional[HyperventilationConfig] = None):
        self.config = config or HyperventilationConfig()
        self.events: list[dict] = []
        self._start_time: Optional[float] = None
        self._safety_acknowledged = False
        self._aborted = False

    @property
    def total_duration_sec(self) -> float:
        """Total paradigm duration in seconds."""
        cycle = (
            self.config.baseline_sec +
            self.config.hyperventilation_sec +
            self.config.recovery_sec
        )
        return cycle * self.config.n_cycles

    def acknowledge_safety(self) -> bool:
        """
        Display safety warnings and require acknowledgment.

        Returns:
            True if safety acknowledged, False if declined.
        """
        print("\n" + "!" * 60)
        print("HYPERVENTILATION SAFETY PROTOCOL")
        print("!" * 60)
        print("""
BEFORE PROCEEDING, CONFIRM THE FOLLOWING:

1. Subject is SEATED with head support (chair with headrest)
2. Partner/observer is PRESENT and can assist if needed
3. Water is within reach
4. Subject understands they can STOP at ANY time

EXPECTED EFFECTS (NORMAL):
- Lightheadedness, mild dizziness
- Tingling in fingers/lips
- Mild visual changes

ABORT IMMEDIATELY IF:
- Sustained dizziness lasting >30 seconds
- SpO2 drops below 92% (if monitoring)
- Numbness/tingling persists after stopping HV
- Chest pain or difficulty breathing
- Any concerning symptoms

Subject and observer have been briefed on safety protocol.
        """)
        print("!" * 60)

        response = input("\nType 'ACKNOWLEDGE' to confirm safety protocol: ").strip()

        if response.upper() == "ACKNOWLEDGE":
            self._safety_acknowledged = True
            logger.info("Safety protocol acknowledged")
            return True
        else:
            logger.warning("Safety protocol NOT acknowledged - paradigm cancelled")
            return False

    def get_schedule(self) -> list[dict]:
        """Generate the paradigm schedule."""
        schedule = []
        t = 0.0

        for cycle in range(self.config.n_cycles):
            # Baseline
            schedule.append({
                "start_sec": t,
                "duration_sec": self.config.baseline_sec,
                "condition": self.CONDITION_BASELINE,
                "label": "baseline",
                "cycle": cycle + 1,
            })
            t += self.config.baseline_sec

            # Hyperventilation
            schedule.append({
                "start_sec": t,
                "duration_sec": self.config.hyperventilation_sec,
                "condition": self.CONDITION_HYPERVENTILATION,
                "label": "hyperventilation",
                "cycle": cycle + 1,
            })
            t += self.config.hyperventilation_sec

            # Recovery
            schedule.append({
                "start_sec": t,
                "duration_sec": self.config.recovery_sec,
                "condition": self.CONDITION_RECOVERY,
                "label": "recovery",
                "cycle": cycle + 1,
            })
            t += self.config.recovery_sec

        return schedule

    def run_interactive(self, callback=None) -> bool:
        """
        Run paradigm interactively with console prompts and breathing cues.

        Args:
            callback: Optional function called at each transition.

        Returns:
            True if completed successfully, False if aborted.
        """
        # Safety check
        if self.config.require_safety_acknowledgment:
            if not self.acknowledge_safety():
                return False

        schedule = self.get_schedule()
        self._start_time = time.time()

        print("\n" + "=" * 50)
        print("HYPERVENTILATION PARADIGM")
        print("=" * 50)
        print(f"Total duration: {self.total_duration_sec / 60:.1f} minutes")
        print("Press Ctrl+C at any time to abort safely")
        print("=" * 50)

        try:
            for block in schedule:
                if self._aborted:
                    break

                condition = block["condition"]
                duration = block["duration_sec"]
                label = block["label"].upper()

                # Countdown
                print(f"\nPreparing for {label}...")
                for sec in range(int(self.config.countdown_sec), 0, -1):
                    print(f"  {sec}...")
                    time.sleep(1)

                _beep()

                # Condition-specific instructions
                print("\n" + "=" * 50)
                if condition == self.CONDITION_BASELINE:
                    print(">>> NORMAL BREATHING <<<")
                    print("Breathe naturally and relax")
                elif condition == self.CONDITION_HYPERVENTILATION:
                    print(">>> HYPERVENTILATION <<<")
                    print(f"Deep breaths at ~{self.config.breathing_rate_per_min:.0f}/min")
                    print("Follow the breathing cue...")
                    print("(You can reduce intensity if uncomfortable)")
                elif condition == self.CONDITION_RECOVERY:
                    print(">>> RECOVERY <<<")
                    print("Return to normal breathing")
                    print("Relax and recover")

                print(f"Duration: {duration / 60:.1f} minutes")
                print("=" * 50)

                self.events.append({
                    "timestamp": time.time() - self._start_time,
                    "event": "condition_start",
                    "condition": condition,
                    "label": block["label"],
                })

                if callback:
                    callback(condition, 0, duration)

                # Run condition with appropriate cues
                if condition == self.CONDITION_HYPERVENTILATION:
                    self._run_hv_phase(duration, callback)
                else:
                    self._run_rest_phase(duration, callback)

                self.events.append({
                    "timestamp": time.time() - self._start_time,
                    "event": "condition_end",
                    "condition": condition,
                })

        except KeyboardInterrupt:
            self._aborted = True
            print("\n\n>>> PARADIGM ABORTED <<<")
            print("Returning to normal breathing...")
            self.events.append({
                "timestamp": time.time() - self._start_time,
                "event": "abort",
                "reason": "user_interrupt",
            })
            _beep()
            time.sleep(1)
            _beep()
            return False

        if not self._aborted:
            print("\n" + "=" * 50)
            print("PARADIGM COMPLETE")
            print("Continue resting for a few minutes before moving")
            print("=" * 50)
            _beep()
            time.sleep(0.2)
            _beep()

        return not self._aborted

    def _run_rest_phase(self, duration: float, callback=None) -> None:
        """Run baseline or recovery phase (no special cues)."""
        start = time.time()
        while time.time() - start < duration:
            elapsed = time.time() - start
            remaining = duration - elapsed

            if int(elapsed) % 60 == 0 and int(elapsed) > 0:
                print(f"  {remaining / 60:.1f} min remaining...")

            if callback:
                callback(None, elapsed, remaining)

            time.sleep(1)

    def _run_hv_phase(self, duration: float, callback=None) -> None:
        """Run hyperventilation phase with breathing cues."""
        breath_interval = 60.0 / self.config.breathing_rate_per_min
        start = time.time()
        last_breath = 0

        while time.time() - start < duration:
            elapsed = time.time() - start
            remaining = duration - elapsed

            # Breathing cue
            if elapsed - last_breath >= breath_interval:
                print("  BREATHE", end="\r", flush=True)
                time.sleep(breath_interval / 2)
                print("         ", end="\r", flush=True)
                last_breath = elapsed

            # Progress every 30 seconds
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                mins_remaining = remaining / 60
                if mins_remaining >= 1:
                    print(f"  {mins_remaining:.1f} min remaining...        ")

            if callback:
                callback(self.CONDITION_HYPERVENTILATION, elapsed, remaining)

            time.sleep(0.1)  # Faster loop for breathing cues

    def label_segments(
        self,
        timestamps: np.ndarray,
        segment_duration_sec: float = 30.0,
    ) -> np.ndarray:
        """Generate condition labels for EEG segments."""
        schedule = self.get_schedule()
        labels = np.full(len(timestamps), -1, dtype=int)

        for i, t in enumerate(timestamps):
            segment_end = t + segment_duration_sec
            for block in schedule:
                block_start = block["start_sec"]
                block_end = block_start + block["duration_sec"]

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
                    "baseline_sec": self.config.baseline_sec,
                    "hyperventilation_sec": self.config.hyperventilation_sec,
                    "recovery_sec": self.config.recovery_sec,
                    "n_cycles": self.config.n_cycles,
                    "breathing_rate": self.config.breathing_rate_per_min,
                },
                "safety_acknowledged": self._safety_acknowledged,
                "aborted": self._aborted,
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


def validate_hv_response(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    sfreq: float = 125.0,
    channel_names: Optional[list[str]] = None,
) -> dict:
    """
    Validate that hyperventilation shows expected slowing (delta/theta increase).

    Args:
        eeg_data: 2D array of shape (n_channels, n_samples).
        labels: Segment labels (0=baseline, 1=hyperventilation, 2=recovery).
        sfreq: Sample rate.
        channel_names: Names for each channel.

    Returns:
        Dict with validation results and power statistics.
    """
    from openbci_eeg.pn_extraction.band_power import compute_band_powers, BandConfig

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(eeg_data.shape[0])]

    results = {
        "slowing_detected": False,
        "delta_theta_ratio_baseline": {},
        "delta_theta_ratio_hv": {},
        "delta_theta_increase": {},
    }

    bands = BandConfig()
    n_channels = eeg_data.shape[0]
    n_samples = eeg_data.shape[1]
    n_segments = len(labels)
    samples_per_segment = n_samples // n_segments

    increases = []

    for ch_idx, ch_name in enumerate(channel_names):
        dt_baseline = []
        dt_hv = []

        for seg_idx, label in enumerate(labels):
            start = seg_idx * samples_per_segment
            end = start + samples_per_segment
            if end > n_samples:
                break

            segment = eeg_data[ch_idx, start:end]
            bp = compute_band_powers(segment, sfreq, bands=bands)
            dt_power = bp["delta"] + bp["theta"]

            if label == 0:  # Baseline
                dt_baseline.append(dt_power)
            elif label == 1:  # Hyperventilation
                dt_hv.append(dt_power)

        mean_baseline = np.mean(dt_baseline) if dt_baseline else 0
        mean_hv = np.mean(dt_hv) if dt_hv else 0
        increase = mean_hv / (mean_baseline + 1e-10)

        results["delta_theta_ratio_baseline"][ch_name] = float(mean_baseline)
        results["delta_theta_ratio_hv"][ch_name] = float(mean_hv)
        results["delta_theta_increase"][ch_name] = float(increase)

        increases.append(increase)

    # Validation: delta+theta should increase by at least 30% during HV
    mean_increase = np.mean(increases)
    results["mean_slowing_ratio"] = float(mean_increase)
    results["slowing_detected"] = mean_increase > 1.3  # 30% increase

    if results["slowing_detected"]:
        logger.info("HV slowing detected: %.1f%% delta+theta increase",
                    (mean_increase - 1) * 100)
    else:
        logger.warning("HV slowing NOT detected: only %.1f%% change",
                       (mean_increase - 1) * 100)

    return results
