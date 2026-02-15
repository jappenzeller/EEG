"""
Evaluation metrics and quantum-classical comparison utilities.

Provides standardized metrics for comparing quantum fidelity-based
classification against classical XGBoost baselines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    accuracy: float
    auc: float
    f1: float
    n_samples: int
    n_features: int
    method: str  # "quantum" or "classical"
    encoding: str  # "V1", "V2", "V3", "V4", "V5", or "classical"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (for AUC). If None, AUC is estimated
            from hard predictions.

    Returns:
        Dict with 'accuracy', 'auc', 'f1', 'precision', 'recall'.
    """
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # AUC
    if y_proba is not None:
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                # Binary classification
                proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics["auc"] = float(roc_auc_score(y_true, proba))
            else:
                # Multiclass
                metrics["auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except ValueError as e:
            logger.warning("Could not compute AUC: %s", e)
            metrics["auc"] = 0.5
    else:
        # Estimate from hard predictions (less accurate)
        metrics["auc"] = (metrics["precision"] + metrics["recall"]) / 2

    return metrics


def compare_quantum_classical(
    quantum_result: EvaluationResult,
    classical_result: EvaluationResult,
) -> dict:
    """
    Compare quantum and classical results with statistical analysis.

    Args:
        quantum_result: Evaluation result from quantum classifier.
        classical_result: Evaluation result from classical classifier.

    Returns:
        Dict with comparison metrics and interpretation.
    """
    comparison = {
        "quantum": {
            "accuracy": quantum_result.accuracy,
            "auc": quantum_result.auc,
            "f1": quantum_result.f1,
            "encoding": quantum_result.encoding,
            "n_params": quantum_result.n_features,
        },
        "classical": {
            "accuracy": classical_result.accuracy,
            "auc": classical_result.auc,
            "f1": classical_result.f1,
            "encoding": classical_result.encoding,
            "n_features": classical_result.n_features,
        },
        "delta": {
            "accuracy": quantum_result.accuracy - classical_result.accuracy,
            "auc": quantum_result.auc - classical_result.auc,
            "f1": quantum_result.f1 - classical_result.f1,
        },
        "feature_ratio": classical_result.n_features / max(quantum_result.n_features, 1),
    }

    # Interpretation based on success criteria from experimental plan
    auc_q = quantum_result.auc
    auc_c = classical_result.auc

    if auc_q > 0.80:
        if auc_q > auc_c:
            interpretation = "ENCODING_SOLVED: Quantum outperforms classical with AUC > 0.80"
        else:
            interpretation = "ENCODING_SOLVED_PARTIAL: High quantum AUC but classical still better"
    elif 0.65 <= auc_q <= 0.80:
        interpretation = "PARTIAL_IMPROVEMENT: Better than baseline but encoding still loses info"
    elif auc_q < 0.65 and auc_c < 0.65:
        interpretation = "PARADIGM_FAILURE: Both methods fail; redesign paradigm"
    else:
        interpretation = "NO_IMPROVEMENT: Quantum underperforms; encoding bottleneck confirmed"

    comparison["interpretation"] = interpretation
    comparison["bottleneck_confirmed"] = (auc_c > 0.80 and auc_q < 0.65)

    return comparison


def channel_scaling_analysis(
    results_by_channels: dict[int, tuple[EvaluationResult, EvaluationResult]],
) -> dict:
    """
    Analyze how performance scales with channel count.

    Compares quantum O(M) vs classical O(M²) complexity scaling.

    Args:
        results_by_channels: Dict mapping n_channels → (quantum_result, classical_result).

    Returns:
        Dict with scaling analysis and crossover point estimation.
    """
    channels = sorted(results_by_channels.keys())

    quantum_aucs = []
    classical_aucs = []

    for n_ch in channels:
        q_result, c_result = results_by_channels[n_ch]
        quantum_aucs.append(q_result.auc)
        classical_aucs.append(c_result.auc)

    quantum_aucs = np.array(quantum_aucs)
    classical_aucs = np.array(classical_aucs)

    # Find crossover point (where quantum >= classical)
    crossover = None
    for i, n_ch in enumerate(channels):
        if quantum_aucs[i] >= classical_aucs[i]:
            crossover = n_ch
            break

    # Compute scaling coefficients (linear fit in log space)
    log_channels = np.log(channels)

    if len(channels) >= 2:
        q_slope = np.polyfit(log_channels, quantum_aucs, 1)[0]
        c_slope = np.polyfit(log_channels, classical_aucs, 1)[0]
    else:
        q_slope = c_slope = 0.0

    return {
        "channels": channels,
        "quantum_aucs": quantum_aucs.tolist(),
        "classical_aucs": classical_aucs.tolist(),
        "crossover_channels": crossover,
        "quantum_slope": float(q_slope),
        "classical_slope": float(c_slope),
        "quantum_advantage_at_16ch": float(quantum_aucs[-1] - classical_aucs[-1])
        if len(channels) > 0 else None,
    }


def generate_report(
    comparison: dict,
    scaling: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate human-readable evaluation report.

    Args:
        comparison: Output from compare_quantum_classical().
        scaling: Optional output from channel_scaling_analysis().
        output_path: If provided, write report to file.

    Returns:
        Report as string.
    """
    lines = [
        "=" * 60,
        "QPNN EEG EVALUATION REPORT",
        "=" * 60,
        "",
        "QUANTUM vs CLASSICAL COMPARISON",
        "-" * 40,
        f"Quantum Encoding: {comparison['quantum']['encoding']}",
        f"  Accuracy: {comparison['quantum']['accuracy']:.3f}",
        f"  AUC:      {comparison['quantum']['auc']:.3f}",
        f"  F1:       {comparison['quantum']['f1']:.3f}",
        f"  Params:   {comparison['quantum']['n_params']}",
        "",
        f"Classical Baseline: {comparison['classical']['encoding']}",
        f"  Accuracy: {comparison['classical']['accuracy']:.3f}",
        f"  AUC:      {comparison['classical']['auc']:.3f}",
        f"  F1:       {comparison['classical']['f1']:.3f}",
        f"  Features: {comparison['classical']['n_features']}",
        "",
        f"Feature Ratio (Classical/Quantum): {comparison['feature_ratio']:.1f}x",
        "",
        "DELTA (Quantum - Classical)",
        f"  Accuracy: {comparison['delta']['accuracy']:+.3f}",
        f"  AUC:      {comparison['delta']['auc']:+.3f}",
        f"  F1:       {comparison['delta']['f1']:+.3f}",
        "",
        f"INTERPRETATION: {comparison['interpretation']}",
        f"Encoding Bottleneck Confirmed: {comparison['bottleneck_confirmed']}",
    ]

    if scaling is not None:
        lines.extend([
            "",
            "CHANNEL SCALING ANALYSIS",
            "-" * 40,
            f"Channels tested: {scaling['channels']}",
            f"Crossover point: {scaling['crossover_channels']} channels",
            f"Quantum slope:   {scaling['quantum_slope']:.4f}",
            f"Classical slope: {scaling['classical_slope']:.4f}",
            f"16-ch advantage: {scaling['quantum_advantage_at_16ch']:+.3f}"
            if scaling['quantum_advantage_at_16ch'] is not None else "",
        ])

    lines.extend(["", "=" * 60])

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info("Report written to %s", output_path)

    return report
