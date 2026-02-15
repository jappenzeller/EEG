"""
A-Gate quantum circuit construction from PN parameters.

Each EEG channel maps to a 2-qubit A-Gate circuit with parameters (a, b, c).
Multi-channel circuits use 2M + 1 qubits for M channels, with a shared
entanglement qubit.

This module wraps QDNU's quantum_agate module when available, and provides
a standalone implementation for when QDNU is not installed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import QDNU; fall back to standalone implementation
try:
    from qdnu.quantum_agate import create_single_channel_agate as _qdnu_agate
    QDNU_AVAILABLE = True
    logger.debug("QDNU quantum_agate module available.")
except ImportError:
    QDNU_AVAILABLE = False
    logger.debug("QDNU not installed; using standalone A-Gate implementation.")


def create_agate_circuit(
    a: float,
    b: float,
    c: float,
    use_qdnu: bool = True,
):
    """
    Create a single-channel A-Gate quantum circuit.

    Args:
        a: Excitatory parameter [0, 1].
        b: Phase parameter [0, 2π].
        c: Inhibitory parameter [0, 1].
        use_qdnu: If True and QDNU is installed, use QDNU's implementation.

    Returns:
        Qiskit QuantumCircuit (2 qubits).

    Raises:
        ImportError: If qiskit is not installed.
    """
    if use_qdnu and QDNU_AVAILABLE:
        return _qdnu_agate(a, b, c)

    # Standalone implementation
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError(
            "qiskit is required for A-Gate circuits. "
            "Install with: pip install openbci-eeg[quantum]"
        )

    qc = QuantumCircuit(2, name=f"A-Gate(a={a:.3f},b={b:.3f},c={c:.3f})")

    # Encode excitatory state on qubit 0
    theta_a = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    qc.ry(theta_a, 0)

    # Encode phase
    qc.rz(b, 0)

    # Encode inhibitory state on qubit 1
    theta_c = 2 * np.arcsin(np.sqrt(np.clip(c, 0, 1)))
    qc.ry(theta_c, 1)

    # Entangle excitatory and inhibitory
    qc.cx(0, 1)

    # Phase interaction
    qc.rz(b * (a - c), 1)

    # Disentangle
    qc.cx(0, 1)

    return qc


def create_multichannel_circuit(
    pn_at_t: dict[str, tuple[float, float, float]],
    channels: Optional[list[str]] = None,
    use_ring_topology: bool = True,
):
    """
    Create multi-channel A-Gate circuit for all channels at a single time point.

    Uses 2M + 1 qubits: 2 per channel + 1 shared entanglement qubit.
    Ring topology CNOT chain connects adjacent channels' excitatory qubits,
    so channel ordering matters — spatially adjacent channels should be
    adjacent in the list to respect nearest-neighbor correlation assumptions.

    Args:
        pn_at_t: Dict mapping channel name → (a, b, c) tuple.
            Typically from pn_extraction.io.pn_at_time().
        channels: Ordered list of channels to include. If None, uses
            RING_ORDER (spatial adjacency) when use_ring_topology=True,
            otherwise dict key order.
        use_ring_topology: If True and channels is None, order channels
            per RING_ORDER for spatially-coherent CNOT chain.

    Returns:
        Qiskit QuantumCircuit with (2 * n_channels + 1) qubits.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("qiskit required. Install with: pip install openbci-eeg[quantum]")

    if channels is None:
        if use_ring_topology:
            from openbci_eeg import RING_ORDER
            # Use ring order, filtered to only channels present in pn_at_t
            channels = [ch for ch in RING_ORDER if ch in pn_at_t]
        else:
            channels = list(pn_at_t.keys())

    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1  # +1 for shared entanglement qubit
    shared_qubit = n_qubits - 1

    qc = QuantumCircuit(n_qubits, name=f"MultiAGate-{n_channels}ch")

    # Initialize shared qubit in superposition
    qc.h(shared_qubit)

    for i, ch_name in enumerate(channels):
        a, b, c = pn_at_t[ch_name]
        q0 = 2 * i       # Excitatory qubit
        q1 = 2 * i + 1   # Inhibitory qubit

        # Per-channel A-Gate encoding
        theta_a = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        theta_c = 2 * np.arcsin(np.sqrt(np.clip(c, 0, 1)))

        qc.ry(theta_a, q0)
        qc.rz(b, q0)
        qc.ry(theta_c, q1)
        qc.cx(q0, q1)
        qc.rz(b * (a - c), q1)
        qc.cx(q0, q1)

        # Entangle with shared qubit
        qc.cx(shared_qubit, q0)

    # Ring topology: CNOT chain between adjacent channels' excitatory qubits
    # This enforces nearest-neighbor spatial correlations in the entanglement
    if n_channels > 1:
        for i in range(n_channels):
            next_i = (i + 1) % n_channels
            qc.cx(2 * i, 2 * next_i)  # Excitatory-to-excitatory ring

    return qc


def compute_fidelity(
    circuit,
    template_statevector: np.ndarray,
) -> float:
    """
    Compute state fidelity between circuit output and a template state.

    Uses statevector simulator for exact fidelity computation.

    Args:
        circuit: Qiskit QuantumCircuit.
        template_statevector: Target state as complex numpy array.

    Returns:
        Fidelity value in [0, 1].
    """
    try:
        from qiskit.quantum_info import Statevector, state_fidelity
    except ImportError:
        raise ImportError("qiskit required for fidelity computation.")

    # Simulate circuit
    sv = Statevector.from_instruction(circuit)

    # Compute fidelity
    template_sv = Statevector(template_statevector)
    return float(state_fidelity(sv, template_sv))
