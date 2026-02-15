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
# Montage: dropped P7/P8 in favor of Pz/Cz for midline ERP coverage
# Rationale: Pz = highest-SNR P300 electrode, Cz = vertex hub for motor/cognitive,
# P7/P8 absent from the paper's 8-ch CHB-MIT baseline so no replication cost.
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T7", "T8",  # Cyton 1-8
    "C3", "C4", "Cz", "Pz", "P3", "P4", "O1", "O2",    # Daisy 9-16
]

# Ring topology ordering for A-Gate CNOT chain.
# Traces a spatial path across the scalp so adjacent channels in the
# entanglement layer are spatially adjacent (nearest-neighbor assumption).
RING_ORDER = [
    "Fp1", "F7", "F3", "C3", "T7", "P3", "O1", "Pz",
    "O2", "P4", "T8", "C4", "F4", "F8", "Fp2", "Cz",
]

# Reference and ground
REFERENCE = "A1/A2"  # Linked earlobes
GROUND = "AFz"       # Forehead midline

# Channel groups for analysis
CHANNEL_GROUPS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8"],
    "temporal": ["T7", "T8"],
    "central": ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
    "midline": ["Cz", "Pz"],
    "p300_primary": ["Pz", "P3", "P4"],
    "gamma_roi": ["F3", "F4", "C3", "C4", "P3", "P4", "Cz"],
    "harmonics_roi": ["C3", "C4", "Cz", "P3", "P4", "Pz"],
}

# Nested channel subsets for Phase 4 scaling experiment.
# Each is a strict subset of the one above it.
# NOTE: 4-channel subset uses left hemisphere chain for spatial contiguity in
# ring topology. Original plan had Fp1/Fp2/F7/F8 but F7→F8 jumps across scalp,
# breaking nearest-neighbor assumption for CNOT chain.
SCALING_SUBSETS = {
    16: RING_ORDER,                                          # Full montage
    8: ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T7", "T8"],  # Matches paper's CHB-MIT
    4: ["Fp1", "F7", "T7", "C3"],                            # Left hemisphere chain
}

# Alternative 4-ch subsets for comparison (non-contiguous but bilateral)
SCALING_SUBSETS_ALT = {
    4: ["Fp1", "Fp2", "F7", "F8"],  # Bilateral frontal (non-contiguous ring)
}
