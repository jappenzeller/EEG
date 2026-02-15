"""
Quantum fidelity-based classification using A-Gate circuits.

The QPNN approach classifies brain states by comparing circuit output states
to pre-computed template states. A segment is classified by which template
state yields higher fidelity.

This module provides:
    - Template state management (creation, storage, loading)
    - Fidelity computation for PN-encoded EEG segments
    - Classification based on template matching
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Template:
    """A template state for fidelity-based classification."""
    name: str
    label: int  # Class label (e.g., 0=baseline, 1=activated)
    statevector: np.ndarray  # Complex statevector
    n_qubits: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FidelityClassifier:
    """
    Classify EEG segments based on quantum state fidelity to templates.

    The classifier:
    1. Encodes EEG segments as PN parameters
    2. Creates A-Gate circuits from PN parameters
    3. Computes fidelity to each template state
    4. Assigns the label of the highest-fidelity template
    """

    def __init__(self, templates: Optional[list[Template]] = None):
        """
        Initialize fidelity classifier.

        Args:
            templates: List of Template objects. Can be added later via add_template().
        """
        self.templates: list[Template] = templates or []
        self._label_map: dict[int, str] = {}

    def add_template(self, template: Template) -> None:
        """Add a template to the classifier."""
        self.templates.append(template)
        self._label_map[template.label] = template.name
        logger.debug("Added template '%s' (label=%d)", template.name, template.label)

    def create_template_from_segments(
        self,
        pn_segments: list[dict[str, dict[str, float]]],
        name: str,
        label: int,
        channel_order: Optional[list[str]] = None,
    ) -> Template:
        """
        Create a template by averaging circuit outputs over multiple segments.

        Args:
            pn_segments: List of PN parameter dicts (one per segment).
                Each dict maps channel_name → {'a', 'b', 'c'}.
            name: Template name.
            label: Class label.
            channel_order: Order of channels for circuit construction.

        Returns:
            Template with averaged statevector.
        """
        from openbci_eeg.bridge.agate import create_multichannel_circuit

        try:
            from qiskit.quantum_info import Statevector
        except ImportError:
            raise ImportError("qiskit required for template creation")

        statevectors = []

        for pn_at_t in pn_segments:
            # Convert to tuple format
            pn_tuples = {ch: (p["a"], p["b"], p["c"]) for ch, p in pn_at_t.items()}

            circuit = create_multichannel_circuit(pn_tuples, channels=channel_order)
            sv = Statevector.from_instruction(circuit)
            statevectors.append(sv.data)

        # Average statevectors (note: this is approximate for quantum states)
        # More rigorous: use density matrix averaging
        mean_sv = np.mean(statevectors, axis=0)
        mean_sv /= np.linalg.norm(mean_sv)  # Renormalize

        n_qubits = int(np.log2(len(mean_sv)))

        template = Template(
            name=name,
            label=label,
            statevector=mean_sv,
            n_qubits=n_qubits,
            metadata={"n_segments": len(pn_segments)},
        )

        self.add_template(template)
        logger.info("Created template '%s' from %d segments", name, len(pn_segments))

        return template

    def compute_fidelities(
        self,
        pn_at_t: dict[str, dict[str, float]],
        channel_order: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Compute fidelity to each template for a single time point.

        Args:
            pn_at_t: PN parameters at one time point.
                Dict mapping channel_name → {'a', 'b', 'c'}.
            channel_order: Order of channels for circuit construction.

        Returns:
            Dict mapping template name → fidelity value.
        """
        from openbci_eeg.bridge.agate import create_multichannel_circuit, compute_fidelity

        # Convert to tuple format
        pn_tuples = {ch: (p["a"], p["b"], p["c"]) for ch, p in pn_at_t.items()}

        circuit = create_multichannel_circuit(pn_tuples, channels=channel_order)

        fidelities = {}
        for template in self.templates:
            fid = compute_fidelity(circuit, template.statevector)
            fidelities[template.name] = fid

        return fidelities

    def predict(
        self,
        pn_at_t: dict[str, dict[str, float]],
        channel_order: Optional[list[str]] = None,
    ) -> tuple[int, dict[str, float]]:
        """
        Predict class label for a single segment.

        Args:
            pn_at_t: PN parameters at one time point.
            channel_order: Order of channels for circuit construction.

        Returns:
            Tuple of (predicted_label, fidelity_dict).
        """
        fidelities = self.compute_fidelities(pn_at_t, channel_order)

        # Find template with highest fidelity
        best_template = max(self.templates, key=lambda t: fidelities[t.name])

        return best_template.label, fidelities

    def predict_batch(
        self,
        pn_segments: list[dict[str, dict[str, float]]],
        channel_order: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for multiple segments.

        Args:
            pn_segments: List of PN parameter dicts.
            channel_order: Order of channels.

        Returns:
            Tuple of (predictions, fidelity_matrix):
                predictions: 1D array of predicted labels
                fidelity_matrix: 2D array of shape (n_segments, n_templates)
        """
        n_segments = len(pn_segments)
        n_templates = len(self.templates)

        predictions = np.zeros(n_segments, dtype=int)
        fidelity_matrix = np.zeros((n_segments, n_templates))

        for i, pn_at_t in enumerate(pn_segments):
            label, fids = self.predict(pn_at_t, channel_order)
            predictions[i] = label
            for j, template in enumerate(self.templates):
                fidelity_matrix[i, j] = fids[template.name]

        return predictions, fidelity_matrix

    def save(self, path: Path) -> None:
        """Save templates to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for template in self.templates:
            # Save statevector as npz
            np.savez(
                path / f"{template.name}.npz",
                statevector_real=template.statevector.real,
                statevector_imag=template.statevector.imag,
            )

            # Save metadata as json
            meta = {
                "name": template.name,
                "label": template.label,
                "n_qubits": template.n_qubits,
                "metadata": template.metadata,
            }
            with open(path / f"{template.name}.json", "w") as f:
                json.dump(meta, f, indent=2)

        logger.info("Saved %d templates to %s", len(self.templates), path)

    def load(self, path: Path) -> None:
        """Load templates from directory."""
        path = Path(path)

        for json_file in path.glob("*.json"):
            with open(json_file) as f:
                meta = json.load(f)

            npz_file = json_file.with_suffix(".npz")
            data = np.load(npz_file)
            statevector = data["statevector_real"] + 1j * data["statevector_imag"]

            template = Template(
                name=meta["name"],
                label=meta["label"],
                statevector=statevector,
                n_qubits=meta["n_qubits"],
                metadata=meta.get("metadata", {}),
            )
            self.add_template(template)

        logger.info("Loaded %d templates from %s", len(self.templates), path)


def compute_template_fidelities(
    pn_timeseries: dict[str, dict[str, np.ndarray]],
    templates: list[Template],
    channel_order: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Compute fidelity to templates over a time series of PN parameters.

    Args:
        pn_timeseries: PN parameters over time.
            Dict mapping channel_name → {'a': array, 'b': array, 'c': array}.
        templates: List of templates to compare against.
        channel_order: Order of channels.

    Returns:
        Dict mapping template_name → fidelity_timeseries (1D array).
    """
    classifier = FidelityClassifier(templates)

    # Get number of time points
    first_channel = next(iter(pn_timeseries.values()))
    n_timepoints = len(first_channel["a"])

    results = {t.name: np.zeros(n_timepoints) for t in templates}

    for t in range(n_timepoints):
        # Extract PN at this time point
        pn_at_t = {
            ch: {"a": params["a"][t], "b": params["b"][t], "c": params["c"][t]}
            for ch, params in pn_timeseries.items()
        }

        fidelities = classifier.compute_fidelities(pn_at_t, channel_order)

        for name, fid in fidelities.items():
            results[name][t] = fid

    return results
