"""
PN parameter serialization and QDNU data contract formatting.

Data contract (what this project exports, QDNU imports):
{
    "metadata": {
        "subject_id": "S001",
        "session_id": "20260131_1430",
        "sample_rate": 125,
        "channels": ["Fp1", "Fp2", ..., "O2"],
        "pn_config": {"lambda_a": 0.1, "lambda_c": 0.05, ...},
    },
    "pn_parameters": {
        "Fp1": {"a": [...], "b": [...], "c": [...]},
        ...
    },
    "timestamps": [...]
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from openbci_eeg.config import PNConfig

logger = logging.getLogger(__name__)


def save_pn_parameters(
    pn_params: dict[str, dict[str, np.ndarray]],
    output_path: str | Path,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save PN parameters to .npz file.

    Args:
        pn_params: Dict keyed by channel name, values are
            {'a': array, 'b': array, 'c': array}.
        output_path: File path (will add .npz if needed).
        metadata: Optional metadata dict (saved as JSON string inside npz).

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten for npz: "Fp1_a", "Fp1_b", "Fp1_c", etc.
    arrays = {}
    for ch_name, params in pn_params.items():
        for param_key in ("a", "b", "c"):
            arrays[f"{ch_name}_{param_key}"] = params[param_key]

    # Store channel list and metadata
    arrays["_channel_names"] = np.array(list(pn_params.keys()))

    if metadata:
        arrays["_metadata"] = np.array(json.dumps(metadata))

    np.savez_compressed(output_path, **arrays)
    logger.info("PN parameters saved to %s", output_path)
    return output_path


def load_pn_parameters(
    path: str | Path,
) -> tuple[dict[str, dict[str, np.ndarray]], Optional[dict]]:
    """
    Load PN parameters from .npz file.

    Returns:
        Tuple of (pn_params dict, metadata dict or None).
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    channel_names = list(data["_channel_names"])

    metadata = None
    if "_metadata" in data:
        metadata = json.loads(str(data["_metadata"]))

    pn_params = {}
    for ch_name in channel_names:
        pn_params[ch_name] = {
            "a": data[f"{ch_name}_a"],
            "b": data[f"{ch_name}_b"],
            "c": data[f"{ch_name}_c"],
        }

    return pn_params, metadata


def pn_to_qdnu_format(
    pn_params: dict[str, dict[str, np.ndarray]],
    subject_id: str = "S000",
    session_id: str = "unknown",
    sample_rate: float = 125.0,
    pn_config: Optional[PNConfig] = None,
    timestamps: Optional[np.ndarray] = None,
) -> dict:
    """
    Convert PN parameters to the QDNU JSON data contract format.

    This is the interchange format between openbci-eeg and QDNU.
    Can be serialized to JSON for S3 upload or passed directly
    to QDNU bridge functions.

    Args:
        pn_params: PN parameters dict.
        subject_id: Subject identifier.
        session_id: Session identifier.
        sample_rate: Original EEG sample rate.
        pn_config: PN config used for extraction.
        timestamps: Optional timestamp array.

    Returns:
        Dict matching the QDNU data contract schema.
    """
    channels = list(pn_params.keys())

    metadata = {
        "subject_id": subject_id,
        "session_id": session_id,
        "sample_rate": sample_rate,
        "channels": channels,
        "n_samples": len(next(iter(pn_params.values()))["a"]),
    }

    if pn_config is not None:
        metadata["pn_config"] = asdict(pn_config)

    # Convert arrays to lists for JSON serialization
    pn_serializable = {}
    for ch_name, params in pn_params.items():
        pn_serializable[ch_name] = {
            "a": params["a"].tolist(),
            "b": params["b"].tolist(),
            "c": params["c"].tolist(),
        }

    result = {
        "metadata": metadata,
        "pn_parameters": pn_serializable,
    }

    if timestamps is not None:
        result["timestamps"] = timestamps.tolist()

    return result


def pn_at_time(
    pn_params: dict[str, dict[str, np.ndarray]],
    time_idx: int,
    channels: Optional[list[str]] = None,
) -> dict[str, tuple[float, float, float]]:
    """
    Extract (a, b, c) tuples at a specific time index across channels.

    This is the format needed to create A-Gate circuits — one (a, b, c)
    tuple per channel at a single time step.

    Args:
        pn_params: Full PN parameters dict.
        time_idx: Sample index.
        channels: Subset of channels. All if None.

    Returns:
        Dict mapping channel name to (a, b, c) float tuple.
    """
    if channels is None:
        channels = list(pn_params.keys())

    result = {}
    for ch in channels:
        p = pn_params[ch]
        result[ch] = (
            float(p["a"][time_idx]),
            float(p["b"][time_idx]),
            float(p["c"][time_idx]),
        )

    return result
