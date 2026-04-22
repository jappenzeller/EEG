"""Optional: export Raw data to foundation model embeddings.

Behind the `foundation-models` extras. Requires separate install:
    pip install -e .[foundation-models]

Clean interface: Raw in, (n_windows, embedding_dim) array out.
Caller supplies the model function; no model weights shipped.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import mne


def embed_raw(
    raw: mne.io.BaseRaw,
    model_fn: Callable[[np.ndarray, float], np.ndarray],
    window_sec: float = 2.0,
    overlap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a model function window-by-window to Raw data.

    Args:
        raw: MNE Raw object
        model_fn: callable(window_data, fs) -> embedding
            window_data: (n_channels, n_window_samples)
            embedding: (embedding_dim,) or (..., embedding_dim)
        window_sec: window length in seconds
        overlap: fractional overlap in [0, 1)

    Returns:
        embeddings: (n_windows, *embedding_dims) array
        centers_sec: (n_windows,) window centers
    """
    data = raw.get_data()
    fs = raw.info["sfreq"]
    n_channels, n_samples = data.shape
    window_samples = int(round(window_sec * fs))
    hop = max(1, int(round(window_samples * (1.0 - overlap))))
    starts = np.arange(0, n_samples - window_samples + 1, hop)

    first_emb = model_fn(data[:, starts[0]:starts[0] + window_samples], fs)
    embeddings = np.zeros(
        (len(starts), *first_emb.shape), dtype=first_emb.dtype,
    )
    embeddings[0] = first_emb
    centers = np.zeros(len(starts), dtype=np.float64)
    centers[0] = (starts[0] + window_samples / 2.0) / fs

    for i, s in enumerate(starts[1:], start=1):
        embeddings[i] = model_fn(data[:, s:s + window_samples], fs)
        centers[i] = (s + window_samples / 2.0) / fs

    return embeddings, centers
