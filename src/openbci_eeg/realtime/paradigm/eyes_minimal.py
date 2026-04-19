"""Minimal 3-block paradigm for alpha blocking validation.

60 s eyes open -> 60 s eyes closed -> 60 s eyes open. Markers at each
transition. Audio tone at each transition.

Marker ID convention:
  150 -- session start
  151 -- eyes open block start
  152 -- eyes closed block start
  153 -- session end
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6 import QtCore


@dataclass
class EyesMinimalPhase:
    label: str
    display_text: str
    duration_sec: float
    marker_id: int


PHASES: list[EyesMinimalPhase] = [
    EyesMinimalPhase("eyes_open_1", "EYES OPEN", 60.0, 151),
    EyesMinimalPhase("eyes_closed", "EYES CLOSED", 60.0, 152),
    EyesMinimalPhase("eyes_open_2", "EYES OPEN", 60.0, 151),
]


class EyesMinimalRunner(QtCore.QObject):
    """Fire markers at phase boundaries. Emit signals for UI updates."""

    phase_changed = QtCore.Signal(str, str, float)  # label, display_text, duration
    tick = QtCore.Signal(float)                      # seconds remaining
    finished = QtCore.Signal()

    def __init__(self, mark_fn: Callable[[int, str], None], parent=None):
        super().__init__(parent)
        self._mark = mark_fn
        self._phase_idx = -1
        self._remaining = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._mark(150, "session_start")
        self._phase_idx = -1
        self._advance()

    def _advance(self):
        self._phase_idx += 1
        if self._phase_idx >= len(PHASES):
            self._mark(153, "session_end")
            self._timer.stop()
            self.finished.emit()
            return
        phase = PHASES[self._phase_idx]
        self._remaining = phase.duration_sec
        self._mark(phase.marker_id, phase.label)
        self.phase_changed.emit(phase.label, phase.display_text, phase.duration_sec)
        self._timer.start()

    def _tick(self):
        self._remaining -= 0.5
        self.tick.emit(max(0.0, self._remaining))
        if self._remaining <= 0.0:
            self._advance()

    def stop(self):
        """Abort without emitting session_end marker."""
        self._timer.stop()
