"""
Template state library for quantum fidelity-based classification.

Templates are reference quantum states representing known brain states
(e.g., normal, pre-ictal, ictal, meditative). Fidelity between a
live circuit's output state and these templates drives classification.

For seizure prediction:
    - Train templates from labeled EEG epochs
    - Compute fidelity against pre-ictal template in sliding window
    - Threshold crossing → seizure warning
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """A reference quantum state for a known brain state."""
    name: str                          # e.g., "normal", "pre_ictal", "ictal"
    statevector: np.ndarray           # Complex state vector
    n_qubits: int = 0
    n_channels: int = 0
    description: str = ""
    source_sessions: list[str] = field(default_factory=list)  # Training session IDs
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.n_qubits == 0:
            self.n_qubits = int(np.log2(len(self.statevector)))


class TemplateLibrary:
    """
    Collection of reference templates for fidelity-based classification.

    Usage:
        lib = TemplateLibrary()
        lib.add("normal", statevector, description="Resting state baseline")
        lib.add("pre_ictal", statevector, description="30s before seizure onset")

        # Classify a new circuit
        scores = lib.score_all(circuit)
        # → {"normal": 0.85, "pre_ictal": 0.12}
    """

    def __init__(self):
        self.templates: dict[str, Template] = {}

    def add(
        self,
        name: str,
        statevector: np.ndarray,
        description: str = "",
        source_sessions: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add or replace a template."""
        self.templates[name] = Template(
            name=name,
            statevector=statevector,
            description=description,
            source_sessions=source_sessions or [],
            metadata=metadata or {},
        )
        logger.info("Template '%s' added (%d qubits).", name, self.templates[name].n_qubits)

    def remove(self, name: str) -> None:
        """Remove a template by name."""
        del self.templates[name]

    def get(self, name: str) -> Template:
        """Get a template by name."""
        return self.templates[name]

    def score_all(self, circuit) -> dict[str, float]:
        """
        Compute fidelity against all templates.

        Args:
            circuit: Qiskit QuantumCircuit.

        Returns:
            Dict mapping template name → fidelity score.
        """
        from openbci_eeg.bridge.agate import compute_fidelity

        scores = {}
        for name, template in self.templates.items():
            scores[name] = compute_fidelity(circuit, template.statevector)
        return scores

    def classify(self, circuit) -> tuple[str, float]:
        """
        Classify circuit by highest-fidelity template.

        Returns:
            Tuple of (template_name, fidelity_score).
        """
        scores = self.score_all(circuit)
        best = max(scores, key=scores.get)
        return best, scores[best]

    @property
    def names(self) -> list[str]:
        return list(self.templates.keys())

    def __len__(self) -> int:
        return len(self.templates)


def save_templates(library: TemplateLibrary, path: str | Path) -> None:
    """Save template library to disk (.npz + .json metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    meta = {}

    for name, template in library.templates.items():
        arrays[f"{name}_sv"] = template.statevector
        meta[name] = {
            "description": template.description,
            "n_qubits": template.n_qubits,
            "n_channels": template.n_channels,
            "source_sessions": template.source_sessions,
            "metadata": template.metadata,
        }

    arrays["_names"] = np.array(list(library.templates.keys()))
    arrays["_meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    logger.info("Template library saved to %s (%d templates).", path, len(library))


def load_templates(path: str | Path) -> TemplateLibrary:
    """Load template library from disk."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    names = list(data["_names"])
    meta = json.loads(str(data["_meta_json"]))

    library = TemplateLibrary()
    for name in names:
        info = meta[name]
        library.add(
            name=name,
            statevector=data[f"{name}_sv"],
            description=info.get("description", ""),
            source_sessions=info.get("source_sessions", []),
            metadata=info.get("metadata", {}),
        )

    logger.info("Loaded template library from %s (%d templates).", path, len(library))
    return library
