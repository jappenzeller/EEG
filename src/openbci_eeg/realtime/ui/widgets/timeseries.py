"""Stacked per-channel time series view.

Renders `window_sec` of the most recent data. Runs purely on the UI thread
via a QTimer. The widget holds a reference to the ring buffer and pulls on
each update -- no push from the acquisition side, which keeps thread
boundaries clean.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg

from ...ring_buffer import RingBuffer

pg.setConfigOptions(antialias=False, useOpenGL=True)


class TimeSeriesWidget(pg.GraphicsLayoutWidget):
    def __init__(
        self,
        ring_buffer: RingBuffer,
        sample_rate: float,
        channel_names: Sequence[str],
        window_sec: float = 4.0,
    ) -> None:
        super().__init__()
        self.setBackground("#0a0e14")
        self.ring = ring_buffer
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.window_samples = int(sample_rate * window_sec)
        self._channel_names = list(channel_names)

        self.plots: list[pg.PlotItem] = []
        self.curves: list[pg.PlotDataItem] = []

        label_style = {"color": "#a0b4c8", "font-size": "9pt"}
        pen = pg.mkPen((74, 158, 255), width=1)

        for i, name in enumerate(self._channel_names):
            p = self.addPlot(row=i, col=0)
            p.setLabel("left", name, **label_style)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=True)
            p.enableAutoRange("y", True)
            p.hideButtons()
            if i < len(self._channel_names) - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            else:
                p.setLabel("bottom", "time (s)", **label_style)
            p.getAxis("left").setWidth(42)
            curve = p.plot(pen=pen)
            self.plots.append(p)
            self.curves.append(curve)

        for p in self.plots[1:]:
            p.setXLink(self.plots[0])

        self._x = np.arange(self.window_samples, dtype=np.float32) / float(sample_rate)

    def update_plot(self) -> None:
        data, _ = self.ring.get_latest(self.window_samples)
        n = data.shape[1]
        if n == 0:
            return
        x = self._x[:n] if n < self.window_samples else self._x
        for i, curve in enumerate(self.curves):
            curve.setData(x, data[i])
