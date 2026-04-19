"""Session HDF5 writer (background thread, chunked append).

Schema:
    /raw                          (n_channels, n_samples)  float32   microvolts
    /timestamps                   (n_samples,)             float64   board time
    /markers/sample_index         (n_markers,)             int64
    /markers/marker_id            (n_markers,)             int32
    /markers/label                (n_markers,)             utf-8
    /feedback/timestamp           (n,)                     float64
    /feedback/state               (n,)                     float32
    /feedback/target              (n,)                     float32   nan if none

Attributes on root:
    sample_rate, n_channels, channel_names, session_uuid,
    start_time_utc, mode, plus anything passed as `metadata`.

Write traffic goes through a queue so the acquisition thread never blocks on
disk. Call `stop()` then `join()` at shutdown to flush cleanly.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

log = logging.getLogger(__name__)


class HDF5Writer(threading.Thread):
    def __init__(
        self,
        path: Path | str,
        n_channels: int,
        sample_rate: int,
        channel_names: list[str],
        metadata: Optional[dict[str, Any]] = None,
        chunk_samples: int = 1000,
    ) -> None:
        super().__init__(daemon=True, name="HDF5Writer")
        self.path = Path(path)
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.channel_names = channel_names
        self.metadata = metadata or {}
        self.chunk_samples = chunk_samples
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._samples_written = 0

    # --- public producer API (callable from any thread) -------------------

    def push_eeg(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        self._queue.put(("eeg", data.copy(), timestamps.copy()))

    def push_marker(self, sample_index: int, marker_id: int, label: str = "") -> None:
        self._queue.put(("marker", int(sample_index), int(marker_id), str(label)))

    def push_feedback(
        self, timestamp: float, state: float, target: Optional[float] = None
    ) -> None:
        self._queue.put(
            ("feedback", float(timestamp), float(state),
             float("nan") if target is None else float(target))
        )

    # --- writer thread ----------------------------------------------------

    def run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log.info("HDF5Writer -> %s", self.path)
        with h5py.File(self.path, "w") as f:
            eeg_ds = f.create_dataset(
                "raw",
                shape=(self.n_channels, 0),
                maxshape=(self.n_channels, None),
                chunks=(self.n_channels, self.chunk_samples),
                dtype=np.float32,
                compression="gzip",
                compression_opts=1,
            )
            ts_ds = f.create_dataset(
                "timestamps",
                shape=(0,),
                maxshape=(None,),
                chunks=(self.chunk_samples,),
                dtype=np.float64,
            )
            markers_grp = f.create_group("markers")
            m_sample_ds = markers_grp.create_dataset(
                "sample_index", shape=(0,), maxshape=(None,), dtype=np.int64
            )
            m_id_ds = markers_grp.create_dataset(
                "marker_id", shape=(0,), maxshape=(None,), dtype=np.int32
            )
            m_label_ds = markers_grp.create_dataset(
                "label",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            fb_grp = f.create_group("feedback")
            fb_ts_ds = fb_grp.create_dataset(
                "timestamp", shape=(0,), maxshape=(None,), dtype=np.float64
            )
            fb_state_ds = fb_grp.create_dataset(
                "state", shape=(0,), maxshape=(None,), dtype=np.float32
            )
            fb_target_ds = fb_grp.create_dataset(
                "target", shape=(0,), maxshape=(None,), dtype=np.float32
            )

            # --- attributes -------------------------------------------------
            f.attrs["sample_rate"] = self.sample_rate
            f.attrs["n_channels"] = self.n_channels
            f.attrs["channel_names"] = np.array(self.channel_names, dtype="S32")
            f.attrs["session_uuid"] = str(uuid.uuid4())
            f.attrs["start_time_utc"] = datetime.now(timezone.utc).isoformat()
            for k, v in self.metadata.items():
                try:
                    f.attrs[k] = v
                except (TypeError, ValueError):
                    f.attrs[k] = str(v)

            # --- drain loop ------------------------------------------------
            while not (self._stop_event.is_set() and self._queue.empty()):
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                kind = item[0]
                try:
                    if kind == "eeg":
                        _, data, ts = item
                        n = data.shape[1]
                        old = eeg_ds.shape[1]
                        eeg_ds.resize((self.n_channels, old + n))
                        eeg_ds[:, old: old + n] = data
                        ts_ds.resize((old + n,))
                        ts_ds[old: old + n] = ts
                        self._samples_written = old + n
                    elif kind == "marker":
                        _, sidx, mid, label = item
                        old = m_sample_ds.shape[0]
                        m_sample_ds.resize((old + 1,))
                        m_sample_ds[old] = sidx
                        m_id_ds.resize((old + 1,))
                        m_id_ds[old] = mid
                        m_label_ds.resize((old + 1,))
                        m_label_ds[old] = label
                    elif kind == "feedback":
                        _, ts, state, target = item
                        old = fb_ts_ds.shape[0]
                        fb_ts_ds.resize((old + 1,))
                        fb_ts_ds[old] = ts
                        fb_state_ds.resize((old + 1,))
                        fb_state_ds[old] = state
                        fb_target_ds.resize((old + 1,))
                        fb_target_ds[old] = target
                except Exception:
                    log.exception("failed to write item: %r", kind)

            f.attrs["end_time_utc"] = datetime.now(timezone.utc).isoformat()
            f.attrs["duration_samples"] = self._samples_written
        log.info("HDF5Writer done (%d samples)", self._samples_written)

    @property
    def samples_written(self) -> int:
        """Samples flushed to disk so far (safe to read from any thread)."""
        return self._samples_written

    def stop(self) -> None:
        self._stop_event.set()
