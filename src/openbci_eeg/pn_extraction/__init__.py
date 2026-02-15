"""
Positive-Negative neuron parameter extraction from EEG signals.

Converts preprocessed EEG data into (a, b, c) PN parameters suitable
for QDNU A-Gate quantum circuits.

Model:
    da/dt = -λ_a · a + f(t)(1 - a)    # Excitatory state
    dc/dt = +λ_c · c + f(t)(1 - c)    # Inhibitory state
    b = phase(f(t))                    # Hilbert phase
"""

from openbci_eeg.pn_extraction.dynamics import (
    extract_pn_single,
    extract_pn_multichannel,
)
from openbci_eeg.pn_extraction.envelope import rms_envelope, normalize_envelope
from openbci_eeg.pn_extraction.io import (
    save_pn_parameters,
    load_pn_parameters,
    pn_to_qdnu_format,
)
from openbci_eeg.pn_extraction.band_power import (
    compute_band_powers,
    compute_alpha_coherence,
    extract_v4_single,
    extract_v4_multichannel,
    extract_v4_windowed,
    BandConfig,
)

__all__ = [
    # V1 encoding (time-domain ODE dynamics)
    "extract_pn_single",
    "extract_pn_multichannel",
    "rms_envelope",
    "normalize_envelope",
    # V4 encoding (multi-scale band power)
    "compute_band_powers",
    "compute_alpha_coherence",
    "extract_v4_single",
    "extract_v4_multichannel",
    "extract_v4_windowed",
    "BandConfig",
    # I/O
    "save_pn_parameters",
    "load_pn_parameters",
    "pn_to_qdnu_format",
]
