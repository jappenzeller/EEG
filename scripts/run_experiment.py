#!/usr/bin/env python3
"""
End-to-end experiment runner for QPNN EEG analysis.

Executes the full pipeline:
    1. Record EEG (or load existing data)
    2. Preprocess signals
    3. Encode with multiple methods (V1-V4)
    4. Evaluate quantum vs classical
    5. Generate comparison report

Usage:
    # Full experiment with recording
    python run_experiment.py --paradigm eyes_open_closed --subject S001

    # Evaluate existing data
    python run_experiment.py --data data/raw/S001_session1/ --evaluate-only

    # Compare all encodings
    python run_experiment.py --data data/processed/S001/ --compare-encodings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openbci_eeg.config import load_config, BoardConfig, PreprocessConfig, PNConfig
from openbci_eeg.preprocessing.filters import preprocess_raw
from openbci_eeg.preprocessing.convert import brainflow_to_mne
from openbci_eeg.pn_extraction.dynamics import extract_pn_multichannel
from openbci_eeg.pn_extraction.band_power import extract_v4_multichannel
from openbci_eeg.evaluation.baseline import (
    extract_classical_features,
    train_xgboost_classifier,
    evaluate_classifier,
)
from openbci_eeg.evaluation.fidelity import FidelityClassifier
from openbci_eeg.evaluation.metrics import (
    compute_metrics,
    compare_quantum_classical,
    channel_scaling_analysis,
    generate_report,
    EvaluationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_recording(
    paradigm: str,
    subject_id: str,
    output_dir: Path,
    synthetic: bool = False,
    config_path: Optional[Path] = None,
) -> Path:
    """
    Record EEG data using specified paradigm.

    Returns path to saved raw data.
    """
    from openbci_eeg.acquisition.board import BoardManager

    logger.info("Starting recording: paradigm=%s, subject=%s", paradigm, subject_id)

    config = load_config(config_path) if config_path else None

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / f"{subject_id}_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Initialize board
    board = BoardManager(synthetic=synthetic, config=config)
    board.connect()

    # Run paradigm
    if paradigm == "eyes_open_closed":
        from openbci_eeg.paradigms.eyes_open_closed import EyesOpenClosedParadigm
        paradigm_runner = EyesOpenClosedParadigm()
    elif paradigm == "hyperventilation":
        from openbci_eeg.paradigms.hyperventilation import HyperventilationParadigm
        paradigm_runner = HyperventilationParadigm()
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")

    # Start recording
    board.start_stream()

    try:
        paradigm_runner.run_interactive()
    finally:
        board.stop_stream()
        data = board.get_data()
        board.disconnect()

    # Save raw data
    np.save(session_dir / "eeg_raw.npy", data)
    paradigm_runner.save_events(session_dir / "events.json")

    # Save metadata
    with open(session_dir / "metadata.json", "w") as f:
        json.dump({
            "subject_id": subject_id,
            "session_id": session_id,
            "paradigm": paradigm,
            "synthetic": synthetic,
            "sample_rate": 125,
            "n_channels": 16,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info("Recording saved to %s", session_dir)
    return session_dir


def run_preprocessing(
    session_dir: Path,
    config: Optional[PreprocessConfig] = None,
) -> Path:
    """
    Preprocess raw EEG data.

    Returns path to processed data.
    """
    logger.info("Preprocessing: %s", session_dir)

    raw_data = np.load(session_dir / "eeg_raw.npy")
    with open(session_dir / "metadata.json") as f:
        metadata = json.load(f)

    sfreq = metadata.get("sample_rate", 125)

    # Convert to MNE
    raw = brainflow_to_mne(raw_data, sfreq)

    # Preprocess
    processed = preprocess_raw(raw, config)

    # Save
    processed_dir = session_dir / "processed"
    processed_dir.mkdir(exist_ok=True)

    processed_data = processed.get_data()
    np.save(processed_dir / "eeg_processed.npy", processed_data)

    logger.info("Preprocessing complete: %s", processed_dir)
    return processed_dir


def run_encoding(
    data_path: Path,
    encodings: list[str] = None,
    segment_sec: float = 30.0,
) -> dict:
    """
    Encode EEG data using multiple methods.

    Returns dict mapping encoding_name → PN parameters.
    """
    if encodings is None:
        encodings = ["V1", "V4"]

    logger.info("Encoding with: %s", encodings)

    # Load data
    if data_path.is_dir():
        eeg_file = data_path / "eeg_processed.npy"
        if not eeg_file.exists():
            eeg_file = data_path / "eeg_raw.npy"
        eeg_data = np.load(eeg_file)
    else:
        eeg_data = np.load(data_path)

    # Load channel names from config or metadata
    from openbci_eeg import CHANNEL_NAMES
    channel_names = CHANNEL_NAMES[:eeg_data.shape[0]]

    sfreq = 125.0  # Default

    results = {}

    for encoding in encodings:
        if encoding == "V1":
            # Time-domain ODE dynamics
            pn = extract_pn_multichannel(eeg_data, sfreq, channel_names)
            results["V1"] = pn
        elif encoding == "V4":
            # Multi-scale band power
            pn = extract_v4_multichannel(eeg_data, sfreq, channel_names)
            results["V4"] = pn
        else:
            logger.warning("Unknown encoding: %s", encoding)

    return results


def run_evaluation(
    pn_data: dict,
    labels: np.ndarray,
    eeg_data: np.ndarray,
    channel_names: list[str],
    encoding_name: str,
) -> tuple[EvaluationResult, EvaluationResult]:
    """
    Evaluate quantum and classical classifiers.

    Returns (quantum_result, classical_result).
    """
    from sklearn.model_selection import train_test_split
    from openbci_eeg import RING_ORDER

    logger.info("Evaluating encoding: %s", encoding_name)

    # Prepare segments
    n_channels, n_samples = eeg_data.shape
    n_segments = len(labels)
    samples_per_segment = n_samples // n_segments

    # Filter valid labels (exclude -1 transition labels)
    valid_mask = labels >= 0
    valid_indices = np.where(valid_mask)[0]
    valid_labels = labels[valid_mask]

    if len(valid_labels) < 10:
        logger.warning("Not enough valid segments for evaluation")
        return None, None

    # --- Classical evaluation ---
    classical_features = []
    for seg_idx in valid_indices:
        start = seg_idx * samples_per_segment
        end = start + samples_per_segment
        segment = eeg_data[:, start:end]
        features, _ = extract_classical_features(segment, channel_names=channel_names)
        classical_features.append(features)

    X_classical = np.array(classical_features)
    y = valid_labels

    X_train, X_test, y_train, y_test = train_test_split(
        X_classical, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = train_xgboost_classifier(X_train, y_train)
    classical_metrics = evaluate_classifier(clf, X_test, y_test)

    classical_result = EvaluationResult(
        accuracy=classical_metrics["accuracy"],
        auc=classical_metrics["auc"],
        f1=classical_metrics["f1"],
        n_samples=len(y),
        n_features=X_classical.shape[1],
        method="classical",
        encoding="XGBoost",
    )

    # --- Quantum evaluation ---
    # Create templates from training data
    train_indices = valid_indices[: int(0.7 * len(valid_indices))]
    test_indices = valid_indices[int(0.7 * len(valid_indices)):]

    # Build segments for each class
    class_0_segments = []
    class_1_segments = []

    for seg_idx in train_indices:
        # Get PN at midpoint of segment
        t_mid = seg_idx * samples_per_segment + samples_per_segment // 2

        # Extract PN values at this time point
        pn_at_t = {}
        for ch_name, ch_pn in pn_data.items():
            if isinstance(ch_pn["a"], np.ndarray):
                t_idx = min(t_mid, len(ch_pn["a"]) - 1)
                pn_at_t[ch_name] = {
                    "a": float(ch_pn["a"][t_idx]),
                    "b": float(ch_pn["b"][t_idx]),
                    "c": float(ch_pn["c"][t_idx]),
                }
            else:
                pn_at_t[ch_name] = ch_pn

        if labels[seg_idx] == 0:
            class_0_segments.append(pn_at_t)
        else:
            class_1_segments.append(pn_at_t)

    # Create fidelity classifier with templates
    fidelity_clf = FidelityClassifier()

    if class_0_segments:
        fidelity_clf.create_template_from_segments(
            class_0_segments, name="class_0", label=0,
            channel_order=[ch for ch in RING_ORDER if ch in channel_names]
        )

    if class_1_segments:
        fidelity_clf.create_template_from_segments(
            class_1_segments, name="class_1", label=1,
            channel_order=[ch for ch in RING_ORDER if ch in channel_names]
        )

    # Evaluate on test set
    test_pn_segments = []
    test_labels = []

    for seg_idx in test_indices:
        t_mid = seg_idx * samples_per_segment + samples_per_segment // 2

        pn_at_t = {}
        for ch_name, ch_pn in pn_data.items():
            if isinstance(ch_pn["a"], np.ndarray):
                t_idx = min(t_mid, len(ch_pn["a"]) - 1)
                pn_at_t[ch_name] = {
                    "a": float(ch_pn["a"][t_idx]),
                    "b": float(ch_pn["b"][t_idx]),
                    "c": float(ch_pn["c"][t_idx]),
                }
            else:
                pn_at_t[ch_name] = ch_pn

        test_pn_segments.append(pn_at_t)
        test_labels.append(labels[seg_idx])

    test_labels = np.array(test_labels)

    if len(fidelity_clf.templates) >= 2:
        predictions, fidelity_matrix = fidelity_clf.predict_batch(test_pn_segments)
        quantum_metrics = compute_metrics(test_labels, predictions, fidelity_matrix)
    else:
        quantum_metrics = {"accuracy": 0.5, "auc": 0.5, "f1": 0.0}

    quantum_result = EvaluationResult(
        accuracy=quantum_metrics["accuracy"],
        auc=quantum_metrics["auc"],
        f1=quantum_metrics["f1"],
        n_samples=len(y),
        n_features=len(channel_names) * 3,  # 3 PN params per channel
        method="quantum",
        encoding=encoding_name,
    )

    return quantum_result, classical_result


def main():
    parser = argparse.ArgumentParser(description="QPNN EEG Experiment Runner")
    parser.add_argument("--paradigm", choices=["eyes_open_closed", "hyperventilation"],
                        help="Paradigm to run")
    parser.add_argument("--subject", default="S001", help="Subject ID")
    parser.add_argument("--data", type=Path, help="Path to existing data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic board")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip recording, evaluate existing data")
    parser.add_argument("--compare-encodings", action="store_true",
                        help="Compare all encoding methods")
    parser.add_argument("--output", type=Path, default=Path("data"),
                        help="Output directory")
    parser.add_argument("--config", type=Path, help="Config file path")

    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Record or load data
    if args.data:
        session_dir = args.data
        logger.info("Using existing data: %s", session_dir)
    elif args.paradigm:
        session_dir = run_recording(
            paradigm=args.paradigm,
            subject_id=args.subject,
            output_dir=output_dir / "raw",
            synthetic=args.synthetic,
            config_path=args.config,
        )
    else:
        parser.error("Either --paradigm or --data is required")

    # Step 2: Preprocess
    processed_dir = run_preprocessing(session_dir)

    # Step 3: Load processed data
    eeg_data = np.load(processed_dir / "eeg_processed.npy")

    with open(session_dir / "events.json") as f:
        events = json.load(f)

    # Generate labels from events
    from openbci_eeg.paradigms.eyes_open_closed import EyesOpenClosedParadigm
    paradigm = EyesOpenClosedParadigm()
    n_samples = eeg_data.shape[1]
    sfreq = 125
    segment_sec = 30.0
    n_segments = int(n_samples / (segment_sec * sfreq))
    timestamps = np.arange(n_segments) * segment_sec
    labels = paradigm.label_segments(timestamps, segment_sec)

    from openbci_eeg import CHANNEL_NAMES
    channel_names = CHANNEL_NAMES[:eeg_data.shape[0]]

    # Step 4: Encode with multiple methods
    if args.compare_encodings:
        encodings = ["V1", "V4"]
    else:
        encodings = ["V4"]  # Default to V4

    encoded_data = run_encoding(processed_dir, encodings)

    # Step 5: Evaluate each encoding
    results = {}
    for encoding_name, pn_data in encoded_data.items():
        q_result, c_result = run_evaluation(
            pn_data, labels, eeg_data, channel_names, encoding_name
        )
        if q_result and c_result:
            results[encoding_name] = (q_result, c_result)

    # Step 6: Generate report
    for encoding_name, (q_result, c_result) in results.items():
        comparison = compare_quantum_classical(q_result, c_result)
        report = generate_report(comparison)
        print(report)

        # Save report
        report_path = session_dir / f"report_{encoding_name}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info("Report saved to %s", report_path)

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
