"""
OpenBCI EEG Acquisition Pipeline for QDNU.

Acquires, preprocesses, and extracts PN parameters from OpenBCI Cyton+Daisy
16-channel EEG data for downstream quantum circuit analysis via QDNU.
"""

__version__ = "0.1.0"

# Hardware constants
BOARD_SAMPLE_RATE = 125  # Hz, 16-channel mode
BOARD_CHANNELS = 16
BOARD_RESOLUTION_BITS = 24
NYQUIST_FREQ = BOARD_SAMPLE_RATE / 2  # 62.5 Hz

# Standard 10-20 channel mapping for 16-channel Cyton+Daisy
# Canonical source of truth: CANONICAL_WIRING.md
# Cyton 1-8: Fp1, Fp2, F3, F4, F7, F8, T7, T8
# Daisy 9-16: C3, C4, P7, P8, P3, P4, O1, O2
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "Oz",  # Cyton 1-8
    "C3", "C4", "Pz", "Cz", "P3", "P4", "O1", "O2",    # Daisy 9-16
]

# Ring topology ordering for A-Gate CNOT chain.
# Traces a spatial path across the scalp so adjacent channels in the
# entanglement layer are spatially adjacent (nearest-neighbor assumption).
# Ring topology ordering for A-Gate CNOT chain.
# Traces a spatial path across the scalp so adjacent channels in the
# entanglement layer are spatially adjacent (nearest-neighbor assumption).
# Updated for canonical montage (P7/P8 replace Pz/Cz).
# Ring topology ordering for A-Gate CNOT chain.
# Full midline spine: Fz -> Cz -> Pz -> Oz
RING_ORDER = [
    "Fp1", "F7", "F3", "Fz", "C3", "P3", "O1", "Oz",
    "Pz", "O2", "P4", "C4", "Cz", "F4", "F8", "Fp2",
]

# Reference and ground (canonical wiring — see CANONICAL_WIRING.md)
# Two independent SRB2 references (no Y-splitter):
#   Cyton SRB2 -> left earlobe (A1)
#   Daisy SRB2 -> right earlobe (A2)
# Two independent BIAS electrodes:
#   Cyton BIAS -> AFz (forehead)
#   Daisy BIAS -> Inion (midline back)
REFERENCE_CYTON = "A1"   # Left earlobe
REFERENCE_DAISY = "A2"   # Right earlobe
BIAS_FRONT = "AFz"       # Forehead midline
BIAS_BACK = "POz"        # Midline posterior (between Pz and Oz)

# Channel groups for analysis
CHANNEL_GROUPS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2", "Oz"],
    "midline": ["Fz", "Cz", "Pz", "Oz"],
    "p300_primary": ["Pz", "P3", "P4", "Cz"],
    "gamma_roi": ["F3", "F4", "C3", "C4", "P3", "P4", "Cz"],
    "alpha_primary": ["O1", "O2", "Oz", "Pz"],
    "fm_theta": ["Fz", "F3", "F4"],
}

# Nested channel subsets for Phase 4 scaling experiment.
# Nested channel subsets for Phase 4 scaling experiment.
SCALING_SUBSETS = {
    16: RING_ORDER,
    8: ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "Oz"],  # Cyton only
    4: ["Fz", "Cz", "Pz", "Oz"],                              # Midline spine
}

SCALING_SUBSETS_ALT = {
    4: ["Fp1", "Fp2", "F7", "F8"],  # Bilateral frontal
}
