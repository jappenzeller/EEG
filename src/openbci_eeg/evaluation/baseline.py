"""
Classical baseline classifiers for head-to-head comparison with quantum encoding.

The paper identified 816 classical features vs 24 PN parameters as a key gap.
This module extracts the classical feature set and trains XGBoost for comparison.

Classical features per channel:
    - Band powers (5): delta, theta, alpha, beta, gamma
    - Band power ratios (10): all pairwise ratios
    - Statistical moments (4): mean, std, skewness, kurtosis
    - Hjorth parameters (3): activity, mobility, complexity
    - Line length (1)
    - Zero crossings (1)
    - Entropy (2): spectral entropy, sample entropy

For 16 channels: 26 features × 16 = 416 channel features
Plus 16×15/2 = 120 pairwise coherence features (alpha band)
Plus cross-channel statistics
Total: ~550-800 features depending on configuration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for classical feature extraction."""
    window_sec: float = 2.0
    include_coherence: bool = True
    include_entropy: bool = True
    bands: dict = None

    def __post_init__(self):
        if self.bands is None:
            self.bands = {
                "delta": (0.5, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 50.0),
            }


def extract_classical_features(
    eeg_data: np.ndarray,
    sfreq: float = 125.0,
    channel_names: Optional[list[str]] = None,
    config: Optional[FeatureConfig] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract classical features from EEG segment.

    Args:
        eeg_data: 2D array of shape (n_channels, n_samples).
        sfreq: Sample rate in Hz.
        channel_names: Names for each channel.
        config: Feature extraction configuration.

    Returns:
        Tuple of (feature_vector, feature_names):
            feature_vector: 1D array of features
            feature_names: List of feature names for interpretability
    """
    if config is None:
        config = FeatureConfig()

    n_channels, n_samples = eeg_data.shape

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]

    features = []
    feature_names = []

    # Per-channel features
    for i, ch_name in enumerate(channel_names):
        ch_data = eeg_data[i]

        # Band powers
        bp = _compute_band_powers(ch_data, sfreq, config.window_sec, config.bands)
        for band_name, power in bp.items():
            features.append(power)
            feature_names.append(f"{ch_name}_{band_name}_power")

        # Band power ratios
        band_names = list(bp.keys())
        for j, b1 in enumerate(band_names):
            for b2 in band_names[j + 1:]:
                ratio = bp[b1] / (bp[b2] + 1e-10)
                features.append(ratio)
                feature_names.append(f"{ch_name}_{b1}_{b2}_ratio")

        # Statistical moments
        features.append(np.mean(ch_data))
        feature_names.append(f"{ch_name}_mean")
        features.append(np.std(ch_data))
        feature_names.append(f"{ch_name}_std")
        features.append(float(skew(ch_data)))
        feature_names.append(f"{ch_name}_skew")
        features.append(float(kurtosis(ch_data)))
        feature_names.append(f"{ch_name}_kurtosis")

        # Hjorth parameters
        activity, mobility, complexity = _hjorth_params(ch_data)
        features.extend([activity, mobility, complexity])
        feature_names.extend([
            f"{ch_name}_hjorth_activity",
            f"{ch_name}_hjorth_mobility",
            f"{ch_name}_hjorth_complexity",
        ])

        # Line length
        ll = np.sum(np.abs(np.diff(ch_data)))
        features.append(ll)
        feature_names.append(f"{ch_name}_line_length")

        # Zero crossings
        zc = np.sum(np.diff(np.sign(ch_data)) != 0)
        features.append(zc)
        feature_names.append(f"{ch_name}_zero_crossings")

        # Entropy measures
        if config.include_entropy:
            spec_ent = _spectral_entropy(ch_data, sfreq, config.window_sec)
            features.append(spec_ent)
            feature_names.append(f"{ch_name}_spectral_entropy")

    # Pairwise coherence (alpha band)
    if config.include_coherence and n_channels > 1:
        from scipy.signal import coherence as scipy_coherence

        alpha_low, alpha_high = config.bands["alpha"]
        nperseg = int(config.window_sec * sfreq)
        nperseg = min(nperseg, n_samples)

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                freqs, coh = scipy_coherence(
                    eeg_data[i], eeg_data[j], fs=sfreq, nperseg=nperseg
                )
                mask = (freqs >= alpha_low) & (freqs < alpha_high)
                if np.any(mask):
                    mean_coh = np.mean(coh[mask])
                else:
                    mean_coh = 0.0
                features.append(mean_coh)
                feature_names.append(f"coh_{channel_names[i]}_{channel_names[j]}")

    logger.debug("Extracted %d classical features", len(features))
    return np.array(features), feature_names


def _compute_band_powers(
    data: np.ndarray,
    sfreq: float,
    window_sec: float,
    bands: dict,
) -> dict[str, float]:
    """Compute relative band powers using Welch PSD."""
    nperseg = int(window_sec * sfreq)
    nperseg = min(nperseg, len(data))

    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)
    total = np.trapezoid(psd, freqs)
    if total == 0:
        total = 1e-10

    result = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        result[name] = float(np.trapezoid(psd[mask], freqs[mask]) / total)

    return result


def _hjorth_params(data: np.ndarray) -> tuple[float, float, float]:
    """Compute Hjorth activity, mobility, and complexity."""
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)

    activity = np.var(data)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)

    return float(activity), float(mobility), float(complexity)


def _spectral_entropy(data: np.ndarray, sfreq: float, window_sec: float) -> float:
    """Compute spectral entropy (normalized Shannon entropy of PSD)."""
    nperseg = int(window_sec * sfreq)
    nperseg = min(nperseg, len(data))

    _, psd = welch(data, fs=sfreq, nperseg=nperseg)

    # Normalize to probability distribution
    psd_norm = psd / (np.sum(psd) + 1e-10)

    # Shannon entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # Normalize by max entropy (log2 of number of frequency bins)
    max_entropy = np.log2(len(psd))
    return float(entropy / max_entropy)


def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
):
    """
    Train XGBoost classifier on classical features.

    Args:
        X_train: Training features of shape (n_samples, n_features).
        y_train: Training labels of shape (n_samples,).
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        random_state: Random seed for reproducibility.

    Returns:
        Trained XGBClassifier instance.

    Raises:
        ImportError: If xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError(
            "xgboost is required for classical baseline. "
            "Install with: pip install xgboost"
        )

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    clf.fit(X_train, y_train)
    logger.info("XGBoost trained: %d samples, %d features", len(y_train), X_train.shape[1])

    return clf


def evaluate_classifier(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate classifier on test set.

    Args:
        clf: Trained classifier with predict_proba method.
        X_test: Test features.
        y_test: True labels.

    Returns:
        Dict with 'accuracy', 'auc', 'f1' metrics.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Handle binary vs multiclass
    if y_proba.shape[1] == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(auc),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
    }
