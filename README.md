# openbci-eeg

EEG acquisition and preprocessing pipeline for the [QDNU](https://github.com/jappenzeller/QDNU) quantum neural analysis project.

Acquires 16-channel EEG from OpenBCI Cyton+Daisy, preprocesses signals, extracts Positive-Negative (PN) neuron parameters, and feeds them into QDNU's A-Gate quantum circuits for seizure prediction and consciousness research.

## Setup

```bash
git clone https://github.com/jappenzeller/openbci-eeg.git
cd openbci-eeg
pip install -e ".[dev]"

# With quantum circuit support
pip install -e ".[dev,quantum]"
```

## Quick Start

```bash
# Verify installation (no hardware needed)
openbci-eeg test-board --synthetic

# Record with synthetic data
openbci-eeg record --synthetic --duration 60 --subject S001

# Record with real hardware
openbci-eeg record --duration 300 --subject S001

# Preprocess
openbci-eeg preprocess data/raw/<session_id>/

# Extract PN parameters
openbci-eeg extract-pn data/processed/<session_id>/
```

## Architecture

```
openbci-eeg/                        QDNU/
├── acquisition  ─── BrainFlow ───► (hardware)
├── preprocessing ── MNE-Python ──► (signal cleaning)
├── pn_extraction ── PN dynamics ──► (a, b, c) parameters
├── bridge ───────── A-Gate ──────► quantum circuits
├── aws ──────────── S3/DynamoDB ─► cloud storage
└── paradigms ────── protocols ───► experiment design
```

## Hardware

| Component | Spec |
|-----------|------|
| Board | OpenBCI Cyton + Daisy |
| Channels | 16 differential |
| Sample Rate | 125 Hz |
| Resolution | 24-bit (ADS1299) |
| Electrodes | Gold cup + Ten20 paste |
| Montage | Standard 10-20 (16 positions) |

## Research Applications

- **P300/ERP:** Oddball paradigm, Sternberg memory task
- **STM/IQ Correlation:** Frank/Lehrl C = S × D model
- **Gamma Oscillations:** Meditation states, 35-45 Hz coherence
- **Seizure Prediction:** PN → A-Gate → template fidelity
- **Higher Harmonics:** Weiss/Frank framework (limited by 62.5 Hz Nyquist)

## Tests

```bash
pytest                    # Full suite (no hardware needed)
pytest --cov=openbci_eeg  # With coverage
```

## License

MIT
