"""Ring buffer tests.

Covers:
    - simple push/read
    - wrap-around
    - multiple pushes aggregating correctly
    - oversized single push (keep tail)
    - read-while-partially-full
"""

import numpy as np
import pytest

from openbci_eeg.realtime.ring_buffer import RingBuffer


def test_push_and_get_exact():
    rb = RingBuffer(n_channels=4, capacity_samples=100)
    data = np.random.randn(4, 50).astype(np.float32)
    ts = np.arange(50, dtype=np.float64)
    rb.push(data, ts)
    out, out_ts = rb.get_latest(50)
    np.testing.assert_array_equal(out, data)
    np.testing.assert_array_equal(out_ts, ts)
    assert rb.total_samples == 50


def test_wrap_around_keeps_most_recent():
    rb = RingBuffer(n_channels=2, capacity_samples=10)
    data = np.arange(30, dtype=np.float32).reshape(2, 15)
    ts = np.arange(15, dtype=np.float64)
    rb.push(data, ts)
    out, out_ts = rb.get_latest(10)
    assert out.shape == (2, 10)
    np.testing.assert_array_equal(out, data[:, 5:15])
    np.testing.assert_array_equal(out_ts, ts[5:15])
    assert rb.total_samples == 15


def test_multiple_pushes_aggregate_in_order():
    rb = RingBuffer(n_channels=2, capacity_samples=100)
    chunks = []
    for i in range(5):
        chunk = np.random.randn(2, 10).astype(np.float32)
        rb.push(chunk, np.arange(i * 10, (i + 1) * 10, dtype=np.float64))
        chunks.append(chunk)
    expected = np.concatenate(chunks, axis=1)
    out, _ = rb.get_latest(50)
    np.testing.assert_array_equal(out, expected)


def test_get_more_than_written_returns_what_is_available():
    rb = RingBuffer(n_channels=2, capacity_samples=100)
    data = np.random.randn(2, 20).astype(np.float32)
    rb.push(data, np.arange(20, dtype=np.float64))
    out, ts = rb.get_latest(50)
    assert out.shape == (2, 20)


def test_oversized_push_is_truncated_to_capacity():
    rb = RingBuffer(n_channels=1, capacity_samples=10)
    data = np.arange(25, dtype=np.float32).reshape(1, 25)
    ts = np.arange(25, dtype=np.float64)
    rb.push(data, ts)
    out, out_ts = rb.get_latest(10)
    np.testing.assert_array_equal(out, data[:, 15:25])
    np.testing.assert_array_equal(out_ts, ts[15:25])
    assert rb.total_samples == 25


def test_shape_mismatch_rejected():
    rb = RingBuffer(n_channels=4, capacity_samples=10)
    with pytest.raises(ValueError):
        rb.push(np.zeros((3, 5), dtype=np.float32), np.arange(5, dtype=np.float64))


def test_empty_push_is_noop():
    rb = RingBuffer(n_channels=2, capacity_samples=10)
    rb.push(np.zeros((2, 0), dtype=np.float32), np.zeros(0, dtype=np.float64))
    assert rb.total_samples == 0
    out, _ = rb.get_latest(10)
    assert out.shape == (2, 0)
