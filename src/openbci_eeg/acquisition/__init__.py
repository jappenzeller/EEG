"""
EEG data acquisition from OpenBCI Cyton+Daisy via BrainFlow.

Supports live hardware, synthetic board (for testing), and file replay.
"""

from openbci_eeg.acquisition.board import connect, disconnect, stream, record
from openbci_eeg.acquisition.synthetic import create_synthetic_board

__all__ = ["connect", "disconnect", "stream", "record", "create_synthetic_board"]
