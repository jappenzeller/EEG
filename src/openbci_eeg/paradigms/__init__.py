"""
Experimental paradigms: stimulus presentation and event marking.

Each paradigm module defines:
    - Stimulus timing and parameters
    - Event codes and markers
    - Recommended electrode subsets
    - Analysis-specific epoch parameters
"""

from openbci_eeg.paradigms.oddball import OddballParadigm
from openbci_eeg.paradigms.sternberg import SternbergParadigm
from openbci_eeg.paradigms.meditation import MeditationProtocol

__all__ = ["OddballParadigm", "SternbergParadigm", "MeditationProtocol"]
