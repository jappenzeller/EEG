"""
Oddball paradigm for P300/ERP research.

Standard auditory or visual oddball:
    - Standard stimuli (80%): frequent, ignored
    - Target stimuli (20%): rare, subject counts or responds
    - ISI: 1-2 seconds
    - Minimum 30 target trials after artifact rejection

Event codes:
    1 = target stimulus onset
    2 = standard stimulus onset
    3 = subject response
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Event codes
EVENT_TARGET = 1
EVENT_STANDARD = 2
EVENT_RESPONSE = 3


@dataclass
class OddballParadigm:
    """
    Oddball paradigm configuration and event generation.

    Attributes:
        n_trials: Total number of trials.
        target_probability: Fraction of target stimuli (default 0.2).
        isi_range: (min, max) inter-stimulus interval in seconds.
        epoch_tmin: Epoch start relative to stimulus (seconds).
        epoch_tmax: Epoch end relative to stimulus (seconds).
        baseline: Baseline correction window.
        priority_channels: Channels most relevant for P300 analysis.
    """
    n_trials: int = 200
    target_probability: float = 0.2
    isi_range: tuple[float, float] = (1.0, 2.0)
    epoch_tmin: float = -0.2
    epoch_tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)
    priority_channels: list[str] = field(
        default_factory=lambda: ["P3", "P4", "C3", "C4"]  # Closest to Pz in our montage
    )

    def generate_trial_sequence(self, seed: Optional[int] = None) -> list[int]:
        """
        Generate randomized trial sequence respecting target probability.

        Ensures no more than 3 consecutive standards between targets
        for adequate target rate.

        Returns:
            List of event codes (EVENT_TARGET or EVENT_STANDARD).
        """
        rng = np.random.default_rng(seed)
        n_targets = int(self.n_trials * self.target_probability)
        n_standards = self.n_trials - n_targets

        sequence = ([EVENT_TARGET] * n_targets + [EVENT_STANDARD] * n_standards)
        rng.shuffle(sequence)
        return sequence

    def generate_isi_times(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random ISIs from uniform distribution.

        Returns:
            Array of ISI values in seconds.
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(self.isi_range[0], self.isi_range[1], size=self.n_trials)

    def generate_onset_samples(
        self,
        sfreq: float = 125.0,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Generate stimulus onset sample indices and event codes.

        Returns:
            Tuple of (onset_samples, event_codes).
        """
        isis = self.generate_isi_times(seed)
        events = self.generate_trial_sequence(seed)

        onset_times = np.cumsum(isis)
        onset_samples = (onset_times * sfreq).astype(int)

        return onset_samples, events

    @property
    def expected_duration_sec(self) -> float:
        """Expected total duration based on mean ISI."""
        mean_isi = sum(self.isi_range) / 2
        return self.n_trials * mean_isi
