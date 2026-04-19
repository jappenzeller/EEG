"""Real-time EEG acquisition, display, and recording.

Provides a Qt-based UI with scrolling time-series, HDF5 session recording,
and a ring-buffer architecture for streaming EEG from OpenBCI hardware.

Entry point:
    python -m openbci_eeg.realtime --mode synthetic --output session.h5
"""

__all__ = ["RingBuffer", "AcquisitionThread", "HDF5Writer", "MainWindow"]
