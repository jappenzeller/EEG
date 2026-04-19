"""Alpha check tab — eyes open / eyes closed paradigm with audio and visual cues.

Shows a large instruction display (EYES OPEN / EYES CLOSED), countdown timer,
progress bar, and plays a tone at each transition. Start button begins the
3-minute paradigm (60s open -> 60s closed -> 60s open).
"""

from __future__ import annotations

import math
import struct
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets, QtMultimedia

from ...paradigm.eyes_minimal import EyesMinimalRunner


# Colors
_BG = QtGui.QColor("#0a0e14")
_OPEN_COLOR = QtGui.QColor("#4a9eff")   # blue for eyes open
_CLOSED_COLOR = QtGui.QColor("#FF453A")  # red for eyes closed
_TEXT_DIM = QtGui.QColor("#889aaa")
_TEXT_BRIGHT = QtGui.QColor("#c8d8e8")


def _generate_tone(freq_hz: float = 880.0, duration_ms: int = 200,
                   sample_rate: int = 44100, volume: float = 0.5) -> bytes:
    """Generate a sine wave tone as raw PCM bytes (16-bit signed, mono)."""
    n_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(n_samples):
        # Apply fade in/out envelope (10ms)
        fade_samples = int(sample_rate * 0.01)
        envelope = 1.0
        if i < fade_samples:
            envelope = i / fade_samples
        elif i > n_samples - fade_samples:
            envelope = (n_samples - i) / fade_samples
        value = volume * envelope * math.sin(2 * math.pi * freq_hz * i / sample_rate)
        samples.append(int(value * 32767))
    return struct.pack(f"<{len(samples)}h", *samples)


class AlphaCheckWidget(QtWidgets.QWidget):
    """Tab widget for the eyes open/closed alpha blocking paradigm."""

    def __init__(
        self,
        mark_fn: Optional[Callable[[int, str], None]] = None,
        get_audio_device: Optional[Callable[[], QtMultimedia.QAudioDevice]] = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, _BG)
        self.setPalette(pal)

        self._mark_fn = mark_fn or (lambda mid, label: None)
        self._get_audio_device = get_audio_device
        self._runner: Optional[EyesMinimalRunner] = None

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Title
        title = QtWidgets.QLabel("Alpha Blocking Check")
        title.setStyleSheet("color: #c8d8e8; font-size: 18pt; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        # Description
        desc = QtWidgets.QLabel(
            "3-minute paradigm: 60s eyes open \u2192 60s eyes closed \u2192 60s eyes open\n"
            "Validates alpha rhythm blocking on occipital channels (O1/O2)."
        )
        desc.setStyleSheet("color: #889aaa; font-size: 11pt;")
        desc.setAlignment(QtCore.Qt.AlignCenter)
        desc.setWordWrap(True)
        main_layout.addWidget(desc)

        main_layout.addStretch(1)

        # Large instruction display
        self._instruction = QtWidgets.QLabel("Ready")
        self._instruction.setAlignment(QtCore.Qt.AlignCenter)
        self._instruction.setStyleSheet(
            "color: #c8d8e8; font-size: 48pt; font-weight: bold;"
        )
        self._instruction.setMinimumHeight(120)
        main_layout.addWidget(self._instruction)

        # Countdown
        self._countdown = QtWidgets.QLabel("")
        self._countdown.setAlignment(QtCore.Qt.AlignCenter)
        self._countdown.setStyleSheet("color: #889aaa; font-size: 24pt;")
        main_layout.addWidget(self._countdown)

        # Progress bar
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 180)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(12)
        self._progress.setStyleSheet("""
            QProgressBar {
                background: #1a2535;
                border: none;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background: #4a9eff;
                border-radius: 6px;
            }
        """)
        main_layout.addWidget(self._progress)

        # Phase indicators (three blocks)
        phase_layout = QtWidgets.QHBoxLayout()
        phase_layout.setSpacing(4)
        self._phase_labels = []
        for text in ["Eyes Open (60s)", "Eyes Closed (60s)", "Eyes Open (60s)"]:
            lbl = QtWidgets.QLabel(text)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(
                "color: #555555; font-size: 10pt; padding: 8px; "
                "background: #12161e; border-radius: 4px;"
            )
            phase_layout.addWidget(lbl)
            self._phase_labels.append(lbl)
        main_layout.addLayout(phase_layout)

        main_layout.addStretch(1)

        # Start/Stop button
        self._btn = QtWidgets.QPushButton("Start Alpha Check (3 min)")
        self._btn.setFixedHeight(50)
        self._btn.setStyleSheet("""
            QPushButton {
                background: #4a9eff; color: white; font-size: 14pt;
                font-weight: bold; border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #3a8eef; }
            QPushButton:disabled { background: #333333; color: #666666; }
        """)
        self._btn.clicked.connect(self._toggle)
        main_layout.addWidget(self._btn)

        # Audio setup
        self._audio_format = QtMultimedia.QAudioFormat()
        self._audio_format.setSampleRate(44100)
        self._audio_format.setChannelCount(1)
        self._audio_format.setSampleFormat(QtMultimedia.QAudioFormat.Int16)

        self._audio_sink: Optional[QtMultimedia.QAudioSink] = None
        self._tone_open = _generate_tone(freq_hz=880.0, duration_ms=150)
        self._tone_closed = _generate_tone(freq_hz=440.0, duration_ms=300)
        self._tone_done = _generate_tone(freq_hz=1320.0, duration_ms=400)

        # State
        self._running = False
        self._total_elapsed = 0.0
        self._current_phase_duration = 0.0
        self._current_phase_idx = -1

    def _play_tone(self, tone_data: bytes) -> None:
        """Play a short tone using QAudioSink on the selected device."""
        try:
            if self._get_audio_device is not None:
                device = self._get_audio_device()
            else:
                device = QtMultimedia.QMediaDevices.defaultAudioOutput()
            if device.isNull():
                return
            self._audio_sink = QtMultimedia.QAudioSink(device, self._audio_format)
            buf = QtCore.QBuffer()
            buf.setData(tone_data)
            buf.open(QtCore.QIODevice.ReadOnly)
            buf.setParent(self._audio_sink)
            self._audio_sink.start(buf)
        except Exception:
            pass  # Audio not critical

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self._running = True
        self._total_elapsed = 0.0
        self._current_phase_idx = -1
        self._btn.setText("Stop")
        self._btn.setStyleSheet("""
            QPushButton {
                background: #FF453A; color: white; font-size: 14pt;
                font-weight: bold; border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #ee3529; }
        """)
        self._progress.setValue(0)

        # Reset phase labels
        for lbl in self._phase_labels:
            lbl.setStyleSheet(
                "color: #555555; font-size: 10pt; padding: 8px; "
                "background: #12161e; border-radius: 4px;"
            )

        self._runner = EyesMinimalRunner(self._mark_fn)
        self._runner.phase_changed.connect(self._on_phase_changed)
        self._runner.tick.connect(self._on_tick)
        self._runner.finished.connect(self._on_finished)
        self._runner.start()

    def _stop(self) -> None:
        if self._runner:
            self._runner.stop()
        self._running = False
        self._instruction.setText("Stopped")
        self._instruction.setStyleSheet(
            "color: #889aaa; font-size: 48pt; font-weight: bold;"
        )
        self._countdown.setText("")
        self._btn.setText("Start Alpha Check (3 min)")
        self._btn.setStyleSheet("""
            QPushButton {
                background: #4a9eff; color: white; font-size: 14pt;
                font-weight: bold; border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #3a8eef; }
        """)

    def _on_phase_changed(self, label: str, display_text: str, duration: float) -> None:
        self._current_phase_idx += 1
        self._current_phase_duration = duration

        is_closed = "closed" in label.lower()

        # Visual cue
        if is_closed:
            color = _CLOSED_COLOR.name()
            self._instruction.setStyleSheet(
                f"color: {color}; font-size: 48pt; font-weight: bold;"
            )
            self._play_tone(self._tone_closed)
        else:
            color = _OPEN_COLOR.name()
            self._instruction.setStyleSheet(
                f"color: {color}; font-size: 48pt; font-weight: bold;"
            )
            self._play_tone(self._tone_open)

        self._instruction.setText(display_text)

        # Highlight active phase indicator
        for i, lbl in enumerate(self._phase_labels):
            if i == self._current_phase_idx:
                bg = "#FF453A" if (i == 1) else "#4a9eff"
                lbl.setStyleSheet(
                    f"color: white; font-size: 10pt; padding: 8px; "
                    f"background: {bg}; border-radius: 4px; font-weight: bold;"
                )
            elif i < self._current_phase_idx:
                lbl.setStyleSheet(
                    "color: #34C759; font-size: 10pt; padding: 8px; "
                    "background: #12161e; border-radius: 4px;"
                )

    def _on_tick(self, remaining: float) -> None:
        self._countdown.setText(f"{remaining:.0f}s")
        elapsed_in_phase = self._current_phase_duration - remaining
        self._total_elapsed = self._current_phase_idx * 60.0 + elapsed_in_phase
        self._progress.setValue(int(self._total_elapsed))

    def _on_finished(self) -> None:
        self._running = False
        self._play_tone(self._tone_done)
        self._instruction.setText("Done")
        self._instruction.setStyleSheet(
            "color: #34C759; font-size: 48pt; font-weight: bold;"
        )
        self._countdown.setText("Session complete")
        self._progress.setValue(180)

        # Mark all phases as done
        for lbl in self._phase_labels:
            lbl.setStyleSheet(
                "color: #34C759; font-size: 10pt; padding: 8px; "
                "background: #12161e; border-radius: 4px;"
            )

        self._btn.setText("Start Alpha Check (3 min)")
        self._btn.setStyleSheet("""
            QPushButton {
                background: #4a9eff; color: white; font-size: 14pt;
                font-weight: bold; border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #3a8eef; }
        """)
