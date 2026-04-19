"""Background thread that polls the board and feeds the ring buffer + callbacks.

Separated from UI thread because BrainFlow calls occasionally block (wireless
retries), and we don't want the UI to freeze. Callbacks are fired on the
acquisition thread -- consumers that touch Qt widgets must marshal to the UI
thread via Qt signals.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from openbci_eeg.acquisition.board import Board
from .ring_buffer import RingBuffer

log = logging.getLogger(__name__)

DataCallback = Callable[[np.ndarray, np.ndarray], None]


class AcquisitionThread(threading.Thread):
    """Poll the board at a fixed interval and push into the ring buffer."""

    def __init__(
        self,
        board: Board,
        ring: RingBuffer,
        poll_interval_sec: float = 0.05,
    ) -> None:
        super().__init__(daemon=True, name="AcquisitionThread")
        self.board = board
        self.ring = ring
        self.poll_interval = poll_interval_sec
        self._stop_event = threading.Event()
        self._callbacks: list[DataCallback] = []

    def add_callback(self, fn: DataCallback) -> None:
        """Add a callback fired with (data, timestamps) on every successful poll.
        Keep callbacks fast and non-blocking (a queue.put is ideal)."""
        self._callbacks.append(fn)

    def run(self) -> None:
        log.info("Acquisition thread started (poll_interval=%.3fs)", self.poll_interval)
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                data, ts = self.board.poll()
                if data.shape[1] > 0:
                    self.ring.push(data, ts)
                    for cb in self._callbacks:
                        try:
                            cb(data, ts)
                        except Exception:
                            log.exception("callback raised; continuing")
            except Exception:
                log.exception("poll failed; continuing")

            elapsed = time.perf_counter() - t0
            self._stop_event.wait(max(0.0, self.poll_interval - elapsed))
        log.info("Acquisition thread stopped")

    def stop(self) -> None:
        self._stop_event.set()
