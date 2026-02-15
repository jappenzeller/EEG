"""
Bridge to QDNU quantum analysis pipeline.

Creates A-Gate quantum circuits from PN parameters and computes
fidelity metrics against template states.

Requires: pip install openbci-eeg[quantum] (installs qiskit)
"""

from openbci_eeg.bridge.agate import (
    create_agate_circuit,
    create_multichannel_circuit,
    compute_fidelity,
)
from openbci_eeg.bridge.templates import (
    TemplateLibrary,
    load_templates,
    save_templates,
)

__all__ = [
    "create_agate_circuit",
    "create_multichannel_circuit",
    "compute_fidelity",
    "TemplateLibrary",
    "load_templates",
    "save_templates",
]
