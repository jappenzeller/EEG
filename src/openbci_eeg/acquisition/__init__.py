"""
EEG data acquisition from OpenBCI Cyton+Daisy via BrainFlow.

Supports live hardware, synthetic board (for testing), and file replay.

Two APIs:
- Functional (CLI / batch): connect(), disconnect(), record(), stream()
- Class-based (realtime): Board, RealtimeBoardConfig, BoardMode
"""

from openbci_eeg.acquisition.board import (
    Board,
    BoardMode,
    RealtimeBoardConfig,
    connect,
    disconnect,
    stream,
    record,
)
from openbci_eeg.acquisition.synthetic import create_synthetic_board

__all__ = [
    "Board",
    "BoardMode",
    "RealtimeBoardConfig",
    "connect",
    "disconnect",
    "stream",
    "record",
    "create_synthetic_board",
]
