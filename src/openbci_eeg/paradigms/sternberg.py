"""
Sternberg memory scanning task for short-term memory / IQ correlation.

Based on Frank/Lehrl C = S × D model:
    C = channel capacity (bits/sec)
    S = processing speed (1 / BIP in sec)
    D = memory span (bits)

Protocol:
    1. Memory set presentation (1-7 items)
    2. Retention interval (2-5 sec)
    3. Probe (in-set or out-of-set)
    4. Subject responds: yes/no
    5. P300 latency increases with set size

Event codes:
    10 = memory set onset
    20 = retention start
    30 = probe onset (in-set)
    31 = probe onset (out-of-set)
    40 = subject response (correct)
    41 = subject response (incorrect)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Event codes
EVENT_MEMORYSET = 10
EVENT_RETENTION = 20
EVENT_PROBE_IN = 30
EVENT_PROBE_OUT = 31
EVENT_RESPONSE_CORRECT = 40
EVENT_RESPONSE_INCORRECT = 41


@dataclass
class SternbergParadigm:
    """
    Sternberg memory scanning task configuration.

    Key measure: P300 latency as a function of set size.
    Linear increase in latency with set size indicates serial scanning.
    Slope relates to processing speed S in the C = S × D model.

    Attributes:
        set_sizes: Memory set sizes to test.
        trials_per_size: Number of trials per set size.
        retention_sec: Retention interval duration.
        probe_in_probability: Probability that probe is in the memory set.
        item_display_sec: Time to display each memory item.
        epoch_tmin: Epoch start relative to probe onset.
        epoch_tmax: Epoch end relative to probe onset.
        priority_channels: Channels for P300 analysis.
    """
    set_sizes: list[int] = field(default_factory=lambda: [1, 3, 5, 7])
    trials_per_size: int = 30
    retention_sec: float = 3.0
    probe_in_probability: float = 0.5
    item_display_sec: float = 0.5
    epoch_tmin: float = -0.2
    epoch_tmax: float = 1.0
    baseline: tuple[float, float] = (-0.2, 0.0)
    priority_channels: list[str] = field(
        default_factory=lambda: ["P3", "P4", "C3", "C4"]
    )

    def generate_trial_block(
        self,
        set_size: int,
        item_pool: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> list[dict]:
        """
        Generate trial specifications for a given set size.

        Args:
            set_size: Number of items in memory set.
            item_pool: Pool of possible items (default: digits 0-9).
            seed: Random seed.

        Returns:
            List of trial dicts with keys:
                memory_set, probe, probe_in_set, expected_response.
        """
        rng = np.random.default_rng(seed)

        if item_pool is None:
            item_pool = list(range(10))

        trials = []
        n_in = int(self.trials_per_size * self.probe_in_probability)

        for i in range(self.trials_per_size):
            memory_set = list(rng.choice(item_pool, size=set_size, replace=False))
            probe_in_set = i < n_in

            if probe_in_set:
                probe = rng.choice(memory_set)
            else:
                remaining = [x for x in item_pool if x not in memory_set]
                probe = rng.choice(remaining)

            trials.append({
                "memory_set": memory_set,
                "set_size": set_size,
                "probe": int(probe),
                "probe_in_set": probe_in_set,
                "expected_response": "yes" if probe_in_set else "no",
            })

        rng.shuffle(trials)
        return trials

    @property
    def total_trials(self) -> int:
        return len(self.set_sizes) * self.trials_per_size

    @property
    def expected_duration_sec(self) -> float:
        """Rough estimate of total experiment duration."""
        per_trial = (
            max(self.set_sizes) * self.item_display_sec  # Memory display
            + self.retention_sec                           # Retention
            + 2.0                                         # Response + ITI
        )
        return self.total_trials * per_trial
