"""Signal quality display -- 16 circles in a 2x8 grid (Cyton / Daisy rows).

Each circle is colored by the worst-of four checks (rail, std, line-noise,
suspicious-quiet), driven by the active Profile.
"""

from __future__ import annotations

from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from ...ring_buffer import RingBuffer
from ...analysis.signal_quality import (
    railed_percent,
    channel_std,
    channel_status,
    line_noise_ratio,
    wire_color,
    channel_position,
    CHANNEL_POSITIONS,
    Profile,
    BENCH,
    STATUS_GREEN,
    STATUS_YELLOW,
    STATUS_RED,
)

_BG = QtGui.QColor("#0a0e14")
_OUTLINE = QtGui.QColor("#1a2535")
_TEXT_PRIMARY = QtGui.QColor("#c8d8e8")
_TEXT_SECONDARY = QtGui.QColor("#889aaa")

_GREEN = QtGui.QColor("#34C759")
_YELLOW = QtGui.QColor("#FFCC00")
_RED = QtGui.QColor("#FF453A")
_GREY = QtGui.QColor("#3C4858")

_STATUS_COLORS = {
    STATUS_GREEN: _GREEN,
    STATUS_YELLOW: _YELLOW,
    STATUS_RED: _RED,
}


def _color_for_status(status: int, has_data: bool) -> QtGui.QColor:
    if not has_data:
        return _GREY
    return _STATUS_COLORS.get(status, _GREY)


class _ChannelCircle(QtWidgets.QWidget):
    """Single channel indicator: label, wire-color ring, status circle, stats."""

    def __init__(
        self,
        channel_number: int,
        name: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.channel_number = channel_number
        self.channel_name = name
        self._wire_color = QtGui.QColor(wire_color(channel_number))
        self._pct: float = 0.0
        self._std: float = 0.0
        self._line: float = 0.0
        self._status: int = STATUS_GREEN
        self._has_data: bool = False
        self.setMinimumSize(55, 120)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

    def set_value(
        self, pct: float, std: float, line: float, status: int, has_data: bool = True
    ) -> None:
        self._pct = pct
        self._std = std
        self._line = line
        self._status = status
        self._has_data = has_data
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        w, h = self.width(), self.height()
        label_h = int(h * 0.14)
        circle_h = int(h * 0.44)
        stats_h = h - label_h - circle_h

        # Channel label
        label_font = QtGui.QFont("monospace", max(7, min(10, w // 8)))
        p.setFont(label_font)
        p.setPen(_TEXT_PRIMARY)
        label_rect = QtCore.QRect(0, 0, w, label_h)
        p.drawText(label_rect, QtCore.Qt.AlignCenter,
                    f"CH{self.channel_number} {self.channel_name}")

        # Circle: wire ring (outer) -> gap -> status fill (inner)
        circle_area = QtCore.QRect(0, label_h, w, circle_h)
        max_radius = (min(circle_area.width(), circle_area.height()) - 4) / 2
        max_radius = max(max_radius, 8)
        ring_width = 4.0
        gap = 2.0
        fill_radius = max(max_radius - ring_width - gap, 6)
        cx = float(circle_area.center().x())
        cy = float(circle_area.center().y())

        # Wire ring
        ring_rect = QtCore.QRectF(
            cx - max_radius, cy - max_radius, max_radius * 2, max_radius * 2)
        p.setPen(QtGui.QPen(self._wire_color, ring_width))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(ring_rect)

        # Status fill
        fill_rect = QtCore.QRectF(
            cx - fill_radius, cy - fill_radius, fill_radius * 2, fill_radius * 2)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(_color_for_status(self._status, self._has_data))
        p.drawEllipse(fill_rect)

        # Stats (three lines)
        stats_font = QtGui.QFont("monospace", max(6, min(8, w // 9)))
        p.setFont(stats_font)
        p.setPen(_TEXT_SECONDARY)
        line_h = stats_h // 3
        stats_top = label_h + circle_h

        if self._has_data:
            texts = [
                f"rail: {self._pct:.0f}%",
                f"std: {self._std:.1f}",
                f"60Hz: {self._line * 100:.0f}%",
            ]
        else:
            texts = ["rail: --", "std: --", "60Hz: --"]

        for j, txt in enumerate(texts):
            r = QtCore.QRect(0, stats_top + j * line_h, w, line_h)
            p.drawText(r, QtCore.Qt.AlignCenter, txt)

        p.end()


class SignalQualityWidget(QtWidgets.QWidget):
    """16-channel signal quality display in a 2x8 grid."""

    def __init__(
        self,
        ring_buffer: RingBuffer,
        sample_rate: float,
        channel_names: Sequence[str],
        profile: Profile = BENCH,
        window_sec: float = 2.0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, _BG)
        self.setPalette(pal)

        self.ring = ring_buffer
        self.sample_rate = sample_rate
        self.profile = profile
        self.window_samples = int(sample_rate * window_sec)

        n_ch = len(channel_names)
        n_cyton = min(8, n_ch)
        n_daisy = max(0, n_ch - 8)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Title with profile name
        title = QtWidgets.QLabel(f"Signal Quality \u2014 {profile.name} profile")
        title.setStyleSheet("color: #c8d8e8; font-size: 14pt; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        main_layout.addStretch(1)

        # Cyton row (CH1-8) — use canonical positions, not BrainFlow names
        cyton_layout = QtWidgets.QHBoxLayout()
        cyton_layout.setSpacing(4)
        cyton_label = QtWidgets.QLabel("CYTON")
        cyton_label.setStyleSheet("color: #889aaa; font-size: 10pt; font-weight: bold;")
        cyton_label.setFixedWidth(52)
        cyton_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        cyton_layout.addWidget(cyton_label)

        self._circles: list[_ChannelCircle] = []
        for ch in range(1, n_cyton + 1):
            circle = _ChannelCircle(ch, CHANNEL_POSITIONS[ch])
            self._circles.append(circle)
            cyton_layout.addWidget(circle)
        main_layout.addLayout(cyton_layout, stretch=4)

        # Daisy row (CH9-16)
        if n_daisy > 0:
            daisy_layout = QtWidgets.QHBoxLayout()
            daisy_layout.setSpacing(4)
            daisy_label = QtWidgets.QLabel("DAISY")
            daisy_label.setStyleSheet("color: #889aaa; font-size: 10pt; font-weight: bold;")
            daisy_label.setFixedWidth(52)
            daisy_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
            daisy_layout.addWidget(daisy_label)

            for ch in range(9, 9 + n_daisy):
                circle = _ChannelCircle(ch, CHANNEL_POSITIONS[ch])
                self._circles.append(circle)
                daisy_layout.addWidget(circle)
            main_layout.addLayout(daisy_layout, stretch=4)

        main_layout.addStretch(1)

        # Dynamic legend from profile thresholds
        p = self.profile
        legend = QtWidgets.QLabel(
            f'<b>Rail:</b> '
            f'<span style="color:#34C759;">\u25CF</span> &lt;{p.rail_green_max_pct:.0f}% ok &nbsp; '
            f'<span style="color:#FFCC00;">\u25CF</span> {p.rail_green_max_pct:.0f}-{p.rail_yellow_max_pct:.0f}% check &nbsp; '
            f'<span style="color:#FF453A;">\u25CF</span> &ge;{p.rail_yellow_max_pct:.0f}% bad'
            f'<br>'
            f'<b>Std:</b> '
            f'<span style="color:#FF453A;">\u25CF</span> &lt;{p.flatline_std_uv:.0f} dead &nbsp; '
            f'<span style="color:#FFCC00;">\u25CF</span> {p.flatline_std_uv:.0f}-{p.low_activity_std_uv:.0f} weak &nbsp; '
            f'<span style="color:#34C759;">\u25CF</span> {p.low_activity_std_uv:.0f}-{p.high_activity_std_uv:.0f} ok &nbsp; '
            f'<span style="color:#FFCC00;">\u25CF</span> {p.high_activity_std_uv:.0f}-{p.saturation_std_uv:.0f} noisy &nbsp; '
            f'<span style="color:#FF453A;">\u25CF</span> &gt;{p.saturation_std_uv:.0f} saturated'
            f'<br>'
            f'<b>60 Hz:</b> '
            f'<span style="color:#34C759;">\u25CF</span> &lt;{p.line_noise_green_max*100:.0f}% ok &nbsp; '
            f'<span style="color:#FFCC00;">\u25CF</span> {p.line_noise_green_max*100:.0f}-{p.line_noise_yellow_max*100:.0f}% check &nbsp; '
            f'<span style="color:#FF453A;">\u25CF</span> &gt;{p.line_noise_yellow_max*100:.0f}% floating'
            f'<br>'
            f'<b>Ring:</b> wire color &nbsp; | &nbsp; '
            f'<b>Sus:</b> std &gt;{p.suspicious_quiet_std_min:.0f} + 60Hz &lt;{p.suspicious_quiet_line_max*100:.0f}% = bias-pinned'
        )
        legend.setStyleSheet("color: #889aaa; font-size: 8pt;")
        legend.setAlignment(QtCore.Qt.AlignCenter)
        legend.setWordWrap(True)
        main_layout.addWidget(legend)

    def update_display(self) -> None:
        """Pull latest data from ring buffer and update circle colors."""
        data, _ = self.ring.get_latest(self.window_samples)
        has_data = data.shape[1] > 0

        if has_data:
            pcts = railed_percent(data, self.profile.rail_threshold_uv)
            stds = channel_std(data)
            lines = line_noise_ratio(data, self.sample_rate)
            statuses = channel_status(pcts, stds, lines, self.profile)
            for i, circle in enumerate(self._circles):
                if i < len(pcts):
                    circle.set_value(
                        float(pcts[i]), float(stds[i]), float(lines[i]),
                        int(statuses[i]), has_data=True,
                    )
        else:
            for circle in self._circles:
                circle.set_value(0.0, 0.0, 0.0, STATUS_GREEN, has_data=False)
