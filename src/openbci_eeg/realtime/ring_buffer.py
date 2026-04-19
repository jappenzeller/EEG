"""Single-writer, multi-reader ring buffer for streaming EEG.

The acquisition thread is the sole writer; readers are the UI, the HDF5
writer, and the feature extractor. A `threading.Lock` serialises push and
read -- at 16 channels x 125 Hz the lock contention is negligible. If you
ever need >1 kHz per channel, move to `multiprocessing.shared_memory` with
atomic index operations.
"""

from __future__ import annotations

import threading
from typing import Tuple

import numpy as np


class RingBuffer:
    """Fixed-capacity circular buffer for multichannel time series."""

    def __init__(self, n_channels: int, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self.n_channels = n_channels
        self.capacity = capacity_samples
        self._buffer = np.zeros((n_channels, capacity_samples), dtype=np.float32)
        self._timestamps = np.zeros(capacity_samples, dtype=np.float64)
        self._write_index = 0  # monotonic total samples written
        self._lock = threading.Lock()

    # --- writer -----------------------------------------------------------

    def push(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """Append samples. data shape (n_channels, n_samples)."""
        if data.ndim != 2 or data.shape[0] != self.n_channels:
            raise ValueError(
                f"data must be ({self.n_channels}, n_samples), got {data.shape}"
            )
        original_n = data.shape[1]
        if original_n == 0:
            return

        # If a single push is larger than capacity, keep only the tail.
        if original_n >= self.capacity:
            data = data[:, -self.capacity:]
            timestamps = timestamps[-self.capacity:]
        n_samples = data.shape[1]

        with self._lock:
            effective_start_index = self._write_index + (original_n - n_samples)
            start = effective_start_index % self.capacity
            end = start + n_samples
            if end <= self.capacity:
                self._buffer[:, start:end] = data
                self._timestamps[start:end] = timestamps
            else:
                first = self.capacity - start
                self._buffer[:, start:] = data[:, :first]
                self._buffer[:, : n_samples - first] = data[:, first:]
                self._timestamps[start:] = timestamps[:first]
                self._timestamps[: n_samples - first] = timestamps[first:]
            self._write_index += original_n

    # --- readers ---------------------------------------------------------

    def get_latest(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (data, timestamps) for the most recent n samples.

        If fewer than n samples have been written, returns what is available.
        """
        with self._lock:
            available = min(self._write_index, self.capacity)
            n = min(n, available)
            if n == 0:
                return (
                    np.zeros((self.n_channels, 0), dtype=np.float32),
                    np.zeros(0, dtype=np.float64),
                )
            end_idx = self._write_index % self.capacity
            start_idx = (self._write_index - n) % self.capacity
            if start_idx < end_idx:
                data = self._buffer[:, start_idx:end_idx].copy()
                ts = self._timestamps[start_idx:end_idx].copy()
            else:
                data = np.concatenate(
                    [self._buffer[:, start_idx:], self._buffer[:, :end_idx]], axis=1
                )
                ts = np.concatenate(
                    [self._timestamps[start_idx:], self._timestamps[:end_idx]]
                )
            return data, ts

    @property
    def total_samples(self) -> int:
        """Total samples written since construction (monotonic)."""
        return self._write_index
