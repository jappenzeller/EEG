"""
Artifact detection and removal: threshold rejection and ICA.
"""

from __future__ import annotations

import logging
from typing import Optional

import mne
import numpy as np
from mne.preprocessing import ICA

from openbci_eeg.config import PreprocessConfig

logger = logging.getLogger(__name__)


def reject_epochs(
    epochs: mne.Epochs,
    threshold_uv: float = 150.0,
) -> mne.Epochs:
    """
    Reject epochs exceeding amplitude threshold.

    Args:
        epochs: MNE Epochs object (must be preloaded).
        threshold_uv: Rejection threshold in µV.

    Returns:
        Epochs with bad epochs dropped.
    """
    threshold_v = threshold_uv * 1e-6  # MNE uses V
    n_before = len(epochs)
    epochs_clean = epochs.copy().drop_bad(reject=dict(eeg=threshold_v))
    n_after = len(epochs_clean)

    logger.info(
        "Artifact rejection: %d/%d epochs kept (threshold=%.0f µV)",
        n_after, n_before, threshold_uv,
    )
    return epochs_clean


def run_ica(
    raw: mne.io.RawArray,
    n_components: int = 15,
    exclude_eog: bool = True,
    random_state: int = 42,
) -> tuple[mne.io.RawArray, ICA]:
    """
    Run ICA for artifact component removal.

    If exclude_eog=True, attempts automatic detection of eye-related
    components using frontal channels (Fp1, Fp2) as EOG proxies.

    Args:
        raw: Preprocessed MNE Raw (should be filtered first).
        n_components: Number of ICA components.
        exclude_eog: Auto-detect and exclude EOG components.
        random_state: For reproducibility.

    Returns:
        Tuple of (cleaned Raw, fitted ICA object).
    """
    ica = ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)
    logger.info("ICA fitted with %d components.", n_components)

    if exclude_eog:
        # Use Fp1/Fp2 as EOG proxies (they pick up blinks)
        eog_channels = [ch for ch in raw.ch_names if ch in ("Fp1", "Fp2")]
        if eog_channels:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw,
                ch_name=eog_channels,
                verbose=False,
            )
            ica.exclude = eog_indices
            logger.info("Auto-excluded EOG components: %s", eog_indices)
        else:
            logger.warning("No Fp1/Fp2 channels found for EOG detection.")

    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean, ica


def detect_bad_channels(
    raw: mne.io.RawArray,
    z_threshold: float = 3.0,
) -> list[str]:
    """
    Detect bad channels based on variance z-score.

    Channels with variance significantly different from the median
    are flagged as bad.

    Args:
        raw: MNE Raw object.
        z_threshold: Z-score threshold for bad channel detection.

    Returns:
        List of bad channel names.
    """
    data = raw.get_data()
    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    mad = np.median(np.abs(variances - median_var))

    if mad == 0:
        return []

    z_scores = 0.6745 * (variances - median_var) / mad  # Robust z-score
    bad_mask = np.abs(z_scores) > z_threshold
    bad_channels = [raw.ch_names[i] for i in np.where(bad_mask)[0]]

    if bad_channels:
        logger.warning("Bad channels detected: %s", bad_channels)

    return bad_channels
