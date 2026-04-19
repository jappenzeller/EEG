# openbci-lattice

Custom Python acquisition and biofeedback pipeline for the OpenBCI Cyton+Daisy, built for the QDNU A-Gate research program.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[audio,dev]
```

## Run (synthetic board — no hardware needed)

```bash
lattice --mode synthetic --output /tmp/session.h5
```

You should see a scrolling time series for 16 synthetic channels. Close the window to end and flush the HDF5 file.

## Run with real hardware

```bash
lattice --mode cyton_daisy --serial-port /dev/ttyUSB0 --output session.h5
```

On macOS the port looks like `/dev/tty.usbserial-XXXXXXXX`. On Windows `COM3` or similar.

## Run from a recorded file (replay)

```bash
lattice --mode playback --playback-file session.h5 --output replay.h5
```

## Test

```bash
pytest
```

## Architecture

```
Board (BrainFlow)  →  RingBuffer  →  TimeSeriesWidget (30 Hz UI update)
       │                    │
       └─ poll 50 ms ──►    └─►  HDF5Writer (background thread, chunked append)
                                 └─►  (future) FeatureExtractor → FeedbackEngine
```

Three threads, strictly separated:
- **Acquisition thread** pulls BrainFlow data every 50 ms, pushes to ring buffer
- **Writer thread** drains a queue and flushes to chunked HDF5
- **UI thread (Qt)** reads ring buffer tail at 30 Hz

The ring buffer is single-writer, multi-reader with a `threading.Lock`. Good enough for 16 ch × 125 Hz; if you push higher (≥1 kHz per channel) switch to `multiprocessing.shared_memory`.

## Session file format

HDF5 with:
- `/raw` — `(n_channels, n_samples)` float32, microvolts, **no filtering applied**
- `/timestamps` — `(n_samples,)` float64, BrainFlow board time
- `/markers/{sample_index, marker_id, label}` — event stream
- `/feedback/{timestamp, state, target}` — feedback state (populated by Phase 6+)

Filtering, referencing, and epoching happen offline in MNE. The display filter path (Phase 3) is **separate** from what's written to disk — raw on disk, filtered for display only.

## Phases

See `PHASES.md` for the Claude Code prompts that extend this scaffold.
