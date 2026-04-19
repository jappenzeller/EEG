"""Main application window for real-time EEG display.

Uses a QTabWidget with:
    Tab 1: Time Series -- scrolling per-channel waveforms
    Tab 2: Signal Quality -- 16 circles showing railed % per channel
    Tab 3: Alpha Check -- eyes open/closed paradigm with audio/visual cues
    Tab 4: Settings -- audio device selection

Only the visible tab is updated each tick to avoid wasted work.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ..ring_buffer import RingBuffer
from ..analysis.signal_quality import Profile, BENCH
from .widgets.timeseries import TimeSeriesWidget
from .widgets.signal_quality import SignalQualityWidget
from .widgets.alpha_check import AlphaCheckWidget
from .widgets.settings import SettingsWidget

_TAB_TIMESERIES = 0
_TAB_SIGNAL_QUALITY = 1
_TAB_ALPHA_CHECK = 2
_TAB_SETTINGS = 3


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        ring_buffer: RingBuffer,
        sample_rate: float,
        channel_names: Sequence[str],
        profile: Profile = BENCH,
        mark_fn: Optional[Callable[[int, str], None]] = None,
        ui_update_hz: int = 30,
    ) -> None:
        super().__init__()
        self.setWindowTitle("openbci-lattice")
        self.resize(1280, 900)

        self.ring = ring_buffer
        self.sample_rate = sample_rate

        # Settings (created first so other widgets can reference it)
        self.settings_widget = SettingsWidget()

        # Widgets
        self.timeseries = TimeSeriesWidget(ring_buffer, sample_rate, channel_names)
        self.signal_quality = SignalQualityWidget(
            ring_buffer, sample_rate, channel_names, profile=profile
        )
        self.alpha_check = AlphaCheckWidget(
            mark_fn=mark_fn,
            get_audio_device=self.settings_widget.get_selected_device,
        )

        # Tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab {
                background: #12161e;
                color: #889aaa;
                padding: 8px 20px;
                border: none;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected {
                color: #c8d8e8;
                border-bottom: 2px solid #4a9eff;
                background: #0a0e14;
            }
        """)
        self.tabs.addTab(self.timeseries, "Time Series")
        self.tabs.addTab(self.signal_quality, "Signal Quality")
        self.tabs.addTab(self.alpha_check, "Alpha Check")
        self.tabs.addTab(self.settings_widget, "Settings")

        self.setCentralWidget(self.tabs)

        # Status bar
        self._status = self.statusBar()
        self._status_label = QtWidgets.QLabel("starting...")
        self._status.addPermanentWidget(self._status_label)

        # Update timer
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / ui_update_hz))

    def _tick(self) -> None:
        idx = self.tabs.currentIndex()
        if idx == _TAB_TIMESERIES:
            self.timeseries.update_plot()
        elif idx == _TAB_SIGNAL_QUALITY:
            self.signal_quality.update_display()

        total = self.ring.total_samples
        self._status_label.setText(
            f"samples: {total:,}    "
            f"elapsed: {total / max(1.0, self.sample_rate):.1f} s"
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)
