"""
Evaluation module: classical baselines and quantum-classical comparison.

Provides XGBoost baseline, fidelity classifier, and head-to-head metrics.
"""

from openbci_eeg.evaluation.baseline import (
    extract_classical_features,
    train_xgboost_classifier,
    evaluate_classifier,
)
from openbci_eeg.evaluation.fidelity import (
    FidelityClassifier,
    compute_template_fidelities,
)
from openbci_eeg.evaluation.metrics import (
    compare_quantum_classical,
    compute_metrics,
)

__all__ = [
    "extract_classical_features",
    "train_xgboost_classifier",
    "evaluate_classifier",
    "FidelityClassifier",
    "compute_template_fidelities",
    "compare_quantum_classical",
    "compute_metrics",
]
