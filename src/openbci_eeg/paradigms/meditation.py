"""
Meditation protocol for gamma oscillation and consciousness studies.

Design:
    1. Baseline (eyes open, rest): 5 min
    2. Baseline (eyes closed, rest): 5 min
    3. Meditation period: 15-30 min
    4. Control (eyes closed, mind-wandering): 15 min
    5. Post-meditation baseline: 5 min

Analysis targets:
    - Gamma power (35-45 Hz) increase during meditation
    - Frontal-parietal gamma coherence
    - Alpha suppression during focused states
    - Comparison across conditions

Event codes:
    100 = eyes open baseline start
    101 = eyes closed baseline start
    102 = meditation start
    103 = meditation end
    104 = mind-wandering start
    105 = mind-wandering end
    106 = post-baseline start
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Event codes
EVENT_BASELINE_OPEN = 100
EVENT_BASELINE_CLOSED = 101
EVENT_MEDITATION_START = 102
EVENT_MEDITATION_END = 103
EVENT_WANDERING_START = 104
EVENT_WANDERING_END = 105
EVENT_POST_BASELINE = 106


@dataclass
class MeditationProtocol:
    """
    Meditation recording protocol configuration.

    Attributes:
        baseline_open_sec: Eyes-open baseline duration.
        baseline_closed_sec: Eyes-closed baseline duration.
        meditation_sec: Meditation period duration.
        mind_wandering_sec: Control condition duration.
        post_baseline_sec: Post-meditation baseline.
        gamma_band: (low, high) Hz for gamma analysis.
        alpha_band: (low, high) Hz for alpha analysis.
        coherence_pairs: Channel pairs for coherence analysis.
        priority_channels: Full montage recommended; these for focused analysis.
    """
    baseline_open_sec: float = 300.0       # 5 min
    baseline_closed_sec: float = 300.0     # 5 min
    meditation_sec: float = 1200.0         # 20 min
    mind_wandering_sec: float = 900.0      # 15 min
    post_baseline_sec: float = 300.0       # 5 min

    gamma_band: tuple[float, float] = (35.0, 45.0)
    alpha_band: tuple[float, float] = (8.0, 12.0)

    coherence_pairs: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("F3", "P3"),   # Left frontal-parietal
            ("F4", "P4"),   # Right frontal-parietal
            ("F3", "F4"),   # Interhemispheric frontal
            ("P3", "P4"),   # Interhemispheric parietal
            ("C3", "C4"),   # Interhemispheric central
        ]
    )

    priority_channels: list[str] = field(
        default_factory=lambda: ["F3", "F4", "C3", "C4", "P3", "P4"]
    )

    def generate_event_timeline(self) -> list[tuple[float, int, str]]:
        """
        Generate ordered list of (time_sec, event_code, description) tuples.

        Returns:
            Timeline of protocol events.
        """
        t = 0.0
        events = []

        events.append((t, EVENT_BASELINE_OPEN, "Eyes-open baseline"))
        t += self.baseline_open_sec

        events.append((t, EVENT_BASELINE_CLOSED, "Eyes-closed baseline"))
        t += self.baseline_closed_sec

        events.append((t, EVENT_MEDITATION_START, "Meditation start"))
        t += self.meditation_sec
        events.append((t, EVENT_MEDITATION_END, "Meditation end"))

        events.append((t, EVENT_WANDERING_START, "Mind-wandering start"))
        t += self.mind_wandering_sec
        events.append((t, EVENT_WANDERING_END, "Mind-wandering end"))

        events.append((t, EVENT_POST_BASELINE, "Post-meditation baseline"))
        t += self.post_baseline_sec

        return events

    @property
    def total_duration_sec(self) -> float:
        return (
            self.baseline_open_sec
            + self.baseline_closed_sec
            + self.meditation_sec
            + self.mind_wandering_sec
            + self.post_baseline_sec
        )

    @property
    def total_duration_min(self) -> float:
        return self.total_duration_sec / 60
