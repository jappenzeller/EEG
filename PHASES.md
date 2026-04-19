# Phased Prompts for Claude Code

Each phase below is a self-contained prompt. Feed them to Claude Code one at a time, in order. Each assumes the previous phases have landed. The scaffold in this repo is Phase 0 — verify it runs (`lattice --mode synthetic --output /tmp/s.h5`) and `pytest` passes before starting Phase 1.

General rules for every phase:

- **Never** apply display-side processing (filters, rereferencing) to the data written to `/raw` in the HDF5 file. `/raw` must remain untouched so offline analysis in MNE starts from truth.
- Callbacks registered on `AcquisitionThread` fire on the acquisition thread. Anything that touches Qt must marshal to the UI thread (Qt signal or `QMetaObject.invokeMethod`).
- BrainFlow exposes one session per process. You cannot run OpenBCI GUI and `lattice` against the same dongle at the same time.
- Every phase must add at least one test or validation script. Scaffold's `tests/` directory is the home.

---

## Phase 1 — PSD widget + signal-quality indicator

**Goal:** Add a frequency-domain view tab and a coarse per-channel signal-quality proxy until Phase 2 gives us real impedance.

**Files to create:**
- `src/openbci_lattice/analysis/__init__.py`
- `src/openbci_lattice/analysis/spectral.py` — `welch_psd(data, fs, nperseg)` wrapper returning `(freqs, psd_db)`
- `src/openbci_lattice/ui/widgets/psd.py` — `PSDWidget` taking the same `ring_buffer`, reading last `2 * fs` samples, updating at 10 Hz
- `tests/test_spectral.py` — sine-wave input at 10 Hz should produce a peak at 10 Hz within 0.5 Hz

**Files to modify:**
- `src/openbci_lattice/ui/main_window.py` — replace the centralWidget with a `QTabWidget` containing `TimeSeriesWidget` and `PSDWidget`

**Design:**
- Welch, 2-second window, 50% overlap, Hann window — gives ~0.5 Hz resolution
- Log-scale y-axis (dB re 1 µV²/Hz), linear x from 0 to `fs/2`
- One line per channel, colored consistently with the time-series tab
- Signal-quality: compute RMS over last 1 s per channel. Map to channel-name label color: green (1–30 µV), yellow (30–100 µV), red (<0.5 µV flatline or >100 µV railed). Expose this in `analysis/spectral.py` as `channel_rms(data)`.

**Acceptance:**
1. `lattice --mode synthetic` shows a PSD tab with broadband content and no NaN/Inf
2. Feeding a known 10 Hz sine into `welch_psd` places the peak at 10 Hz ±0.5 Hz (unit test)
3. Tabs switch without freezing the UI
4. PSDWidget does not call into BrainFlow directly — it reads the ring buffer only

---

## Phase 2 — Impedance check via ADS1299 lead-off

**Goal:** Replicate OpenBCI GUI's impedance check using BrainFlow `config_board()` with the ADS1299 lead-off current (6 nA at 31.25 Hz).

**Files to create:**
- `src/openbci_lattice/acquisition/impedance.py` — `ImpedanceRunner` that sequences per-channel lead-off enable/measure/disable, returns `dict[channel_index → kΩ]`
- `src/openbci_lattice/ui/widgets/impedance_dialog.py` — modal with a bar chart (PyQtGraph `BarGraphItem`), threshold line at 10 kΩ, color-coded bars (<5 green, 5–10 yellow, 10–20 orange, >20 red)
- `tests/test_impedance.py` — unit-test the command-string builder only (actual measurement needs hardware)

**Files to modify:**
- `src/openbci_lattice/ui/main_window.py` — add toolbar button "Check impedance"

**Design:**
- Command format (Cyton firmware): for channel `N` (1-indexed 1–16), send `z<N><Plead><Nlead>Z`. Enable with `z<N>101Z` (N-side lead-off on), disable with `z<N>000Z`. See OpenBCI docs/board_commands.md
- Channels 9–16 live on the Daisy — same syntax, the dongle routes it
- Measurement: enable lead-off, wait 500 ms settle, pull 2 s of data via `board.poll()`, extract 31.25 Hz amplitude via Goertzel filter (preferred — minimal FFT leakage) or interpolated FFT peak. Impedance = `V_peak / 6e-9`
- **Critical:** disable lead-off on every channel before returning control, even if the measurement fails. Enabled lead-off makes the EEG unusable
- Run channels sequentially, not in parallel — the 31.25 Hz current cross-talks if multiple channels are active
- In SYNTHETIC mode, skip the hardware path and return simulated values (e.g. uniform random 3–15 kΩ) so the UI flow is testable

**Acceptance:**
1. Live-board run measures all 16 channels in ≤60 s and displays a bar chart
2. Data streaming resumes without any artifact or dropped samples after the dialog closes
3. Synthetic-mode path produces a plausible dialog without touching `config_board()`
4. Unit test verifies command strings match the Cyton firmware spec exactly

---

## Phase 3 — Real-time filter pipeline

**Goal:** Two separate filter paths. Display filter (low-latency IIR) for what goes on screen; zero-phase for offline. Raw disk data is **never** filtered.

**Files to create:**
- `src/openbci_lattice/preprocessing/filters.py` — `StreamingIIR` class wrapping `scipy.signal.sosfilt_zi`/`sosfilt` with per-channel state; `design_butter_bandpass(low, high, fs, order=4)`; `design_notch(freq, fs, Q=30)`
- `tests/test_filters.py` — verify a step response, verify state persistence across chunked application matches a single-shot filter

**Files to modify:**
- `src/openbci_lattice/ui/widgets/timeseries.py` — optional `display_filter: StreamingIIR | None` param, applied to the data read from the ring before plotting
- `src/openbci_lattice/ui/main_window.py` — toolbar controls: low-cut (spinbox, 0–5 Hz), high-cut (spinbox, 20–62 Hz), notch toggle + 50/60 Hz selector. Rebuilding filters clears state (fine for display use)

**Design:**
- Cascade: bandpass SOS, then notch SOS. Apply with `sosfilt` maintaining per-channel `zi`
- Reset `zi` to `sosfilt_zi(...) * first_sample` on first call to avoid a big initial transient
- Filter applied only to what the *UI* reads. The HDF5 writer still gets the raw data directly from `AcquisitionThread.add_callback` (which fires with raw)
- Do not add an "offline filter" API in this phase — MNE does that job when the HDF5 is loaded later

**Acceptance:**
1. DC drift visible in synthetic mode disappears when bandpass is enabled
2. 60 Hz line noise (inject via synthetic or real board) is attenuated ≥30 dB in PSD when notch is on
3. HDF5 produced from a filtered-display session, reloaded and PSD-analyzed, shows the un-attenuated 60 Hz (proves disk data is raw)
4. Unit test: chunked filtering with state carry-over produces the same output as single-shot filtering (within float32 tolerance)

---

## Phase 4 — Markers, session metadata, recording control

**Goal:** Give the user a way to annotate the stream with sample-accurate markers, set session metadata at the start, and start/stop recording independently of the UI window lifetime.

**Files to create:**
- `src/openbci_lattice/acquisition/markers.py` — `MarkerLog` that accumulates `(sample_index, marker_id, label, wall_time)` tuples in memory; uses `ring.total_samples` as the sample index at insertion time
- `src/openbci_lattice/ui/widgets/session_control.py` — dock widget with: Record toggle, subject-id field, paradigm dropdown, notes text area, configurable marker buttons (default: "Eyes Open", "Eyes Closed", "Blink", "Artifact")
- `src/openbci_lattice/ui/widgets/session_dialog.py` — modal shown at startup to fill subject / paradigm / notes
- `tests/test_markers.py` — sample-index alignment test using a synthetic known stream

**Files to modify:**
- `src/openbci_lattice/__main__.py` — show session dialog before instantiating the writer; only start the writer when "Record" is toggled on (not at app launch)
- `src/openbci_lattice/ui/main_window.py` — register global shortcuts 1–9 for the first 9 configured markers; wire marker buttons to both `board.insert_marker()` AND `writer.push_marker()`

**Design:**
- Prefer `board.insert_marker(value)` for truly sample-accurate marks — BrainFlow writes them into the marker channel in the same sample clock as EEG. `writer.push_marker()` is the redundant path that uses `ring.total_samples` as the index; compare the two offline to quantify jitter
- `ring.total_samples` is monotonic across `RingBuffer` instances within a session — pair with `writer.samples_written` to detect dropped polls
- Marker ID space: reserve 1–99 for manual markers, 100–199 for paradigm-phase transitions (Phase 5), 200+ for feedback events (Phase 6–8)

**Acceptance:**
1. Pressing "Eyes Closed" during a recording produces a visible marker entry in `/markers/*` in the HDF5 with the correct label
2. Keybinding "2" inserts the second configured marker from any widget focus
3. Session metadata (subject, paradigm, notes) appears as attrs on the HDF5 root
4. Recording toggle off → on → off produces a single continuous HDF5 per toggle cycle with no samples from the "off" period

---

## Phase 5 — Paradigm runner + eyes-open/closed paradigm

**Goal:** State-machine-based paradigm execution: a paradigm is a list of phases, each with `(duration_sec, label, marker_id, instruction_text)`. The runner inserts markers at each transition, updates an on-screen instruction overlay, and runs a countdown.

**Files to create:**
- `src/openbci_lattice/paradigm/base.py` — `Paradigm` dataclass + `ParadigmRunner` (a `QObject` with signals: `phase_started(PhaseInfo)`, `tick(seconds_remaining)`, `finished()`); internally uses `QTimer`
- `src/openbci_lattice/paradigm/eyes_open_closed.py` — the exact paradigm from `eeg_experiment_plan.md` Paradigm A: 5 min open → 5 min closed, repeat 3×
- `src/openbci_lattice/ui/widgets/paradigm_overlay.py` — large semi-transparent overlay widget with instruction text + countdown
- `tests/test_paradigm.py` — with time mocked, verify the phase sequence and that every transition emits a marker

**Files to modify:**
- `src/openbci_lattice/ui/main_window.py` — paradigm dropdown + Start/Pause/Stop in the toolbar; overlay as a sibling of the central widget

**Design:**
- The runner owns the state machine. It does **not** own the acquisition or writer — it calls into them via callbacks/signals
- Pause must not stop acquisition — only the phase timer. "Stop" emits a final marker and resets
- Keep paradigms data-driven enough that Phase 8 can define a longer multi-block paradigm without rewriting the framework
- YAML-as-source is overkill for now — paradigms are just Python dataclasses

**Acceptance:**
1. Running the eyes-open/closed paradigm displays correct instruction text at each transition
2. The HDF5 contains exactly 7 paradigm markers (1 start + 6 transitions) with IDs in the 100–199 range and the right labels
3. Pausing mid-phase then resuming preserves the remaining duration
4. Unit test: mock `QTimer` to fire immediately, verify the phase sequence and marker emissions

---

## Phase 6 — Audio feedback engine

**Goal:** Low-latency audio output with amplitude-modulated pink noise and event chimes. Hardware-selectable output device. Logs every event as a marker so auditory ERPs can be subtracted offline.

**Files to create:**
- `src/openbci_lattice/feedback/audio/engine.py` — `AudioEngine` wrapping `sounddevice.OutputStream` with a callback that reads a shared `numpy.float32` amplitude value. Supports multiple simultaneous sources (pink noise + one-shot chimes mixed)
- `src/openbci_lattice/feedback/audio/generators.py` — `pink_noise_generator(fs)` (Voss–McCartney or FIR), `chime(freq, duration, envelope)`
- `src/openbci_lattice/feedback/audio/settings.py` — dataclass storing: output device index, volume limit, modality label (`"speakers"` / `"earbuds_wired"` / `"earbuds_bluetooth"`). The label is written to HDF5 attrs every session
- `tests/test_audio_generators.py` — pink noise PSD 1/f slope ≈ -1 within tolerance; chime envelope shape is correct

**Files to modify:**
- `src/openbci_lattice/ui/widgets/session_control.py` — audio device dropdown (`sounddevice.query_devices`), modality selector, test-tone button
- `src/openbci_lattice/io/hdf5_writer.py` — no schema change; feedback module writes via `push_feedback` and `push_marker` already

**Design:**
- Callback-based `OutputStream`, blocksize 256 @ 44.1 kHz = ~6 ms buffer
- All audio parameters (amplitude, pitch) are `numpy.float32` scalars in a shared `multiprocessing.sharedctypes` or just plain module-level locked values — reading in the audio callback must not call into the GIL path
- Every chime insertion calls `board.insert_marker(250.0)` (reserved range 200–299) and `writer.push_marker()`. Offline: regress auditory ERPs around these timestamps before drawing conclusions about other effects
- **Never** use `pygame.mixer` — its scheduler jitter is ~20–50 ms and unacceptable
- UI warning text when the subject selects "earbuds_bluetooth" or when the paradigm is an ERP paradigm and the modality is "speakers" (latency jitter issue for both)

**Acceptance:**
1. Pink noise plays; amplitude scalar changes produce audible volume changes within ≤30 ms (wall-clock measurable with loopback)
2. Every chime inserts a marker in both BrainFlow marker stream and `/markers` group
3. Switching output device does not crash or leak streams
4. Modality label appears in HDF5 `audio_modality` root attr
5. `pytest tests/test_audio_generators.py` passes

---

## Phase 7 — Bivariate 2D visual feedback widget

**Goal:** Real-time scatter showing two features on X/Y with a trailing history and optional target region. Used by Phase 8.

**Files to create:**
- `src/openbci_lattice/feedback/features.py` — `StreamingFeatureExtractor` base class: `update(new_data, new_timestamps) -> float`. First concrete implementation: `BandPowerExtractor(channels, band=(8, 12), window_sec=1.0, hop_sec=0.25)` — maintains a ring of the latest window, computes band power via Welch on every hop
- `src/openbci_lattice/feedback/visual/bivariate.py` — `BivariateFeedbackWidget`: takes two `StreamingFeatureExtractor` instances and a target region definition. Renders dot + trail (last 30 s) + target region
- `src/openbci_lattice/feedback/visual/mapping.py` — `Mapping` class: linear / log / target-relative transforms from feature value → plotted coordinate. Use `log` for power features (they span decades)
- `tests/test_features.py` — feed known sines, verify extractors return correct power

**Files to modify:**
- `src/openbci_lattice/ui/main_window.py` — add feedback tab to the `QTabWidget`
- `src/openbci_lattice/acquisition/acquisition_thread.py` — `add_callback` already supports this; feature extractors register as callbacks

**Design:**
- Feature extractors run on the acquisition thread (fast enough — Welch on 1 s × 16 ch ≈ 2 ms)
- They push values to the widget via a Qt signal or `QMetaObject.invokeMethod` — again, no direct GUI calls from non-UI threads
- Target region: pass a polygon or ellipse `QPainterPath`; check `path.contains(point)` per update; emit a signal when entering/leaving so Phase 8 can use it
- **No smoothing.** If the feature is too noisy to display raw, lengthen the window, don't EMA the output — that decouples the visual from the underlying signal and defeats the purpose of the feedback loop

**Acceptance:**
1. Dot tracks a test signal (eyes-open/closed alpha) in real time with ≤300 ms perceived lag (window + hop + render)
2. Trail smoothly fades over 30 s
3. Switching features (e.g. frontal beta vs posterior alpha) clears the trail and updates the axes
4. Target region triggers a visual cue (halo or color change) on entry

---

## Phase 8 — Alpha-training paradigm with feedback-on / feedback-off control

**Goal:** The first real research-usable paradigm. Integrates Phases 4–7 into a session with a proper control condition.

**Block structure:**
1. 2 min baseline (eyes closed, no feedback, recording on)
2. 10 min feedback-on (eyes closed, posterior alpha power → inverse pink-noise amplitude: more alpha → quieter)
3. 2 min rest (eyes open)
4. 10 min feedback-off control (eyes closed, no audio, no visual — identical instruction to block 2 without the contingency)
5. 2 min baseline

**Files to create:**
- `src/openbci_lattice/paradigm/alpha_training.py` — the paradigm; registers feature extractors and audio engine per-block; swaps feedback off during the control block by zeroing the amplitude mapping
- `tests/test_alpha_training.py` — with time mocked, verify the markers, feedback-state log, and the fact that `/feedback/state` is continuous during block 2 and NaN during block 4

**Files to modify:**
- `src/openbci_lattice/paradigm/base.py` — extend `PhaseInfo` with optional `on_enter` / `on_exit` hooks so feedback can be enabled/disabled per phase
- `src/openbci_lattice/io/hdf5_writer.py` — no schema change

**Design:**
- Posterior alpha = mean power across channels 15, 16, and Cz (if present; otherwise just O1/O2)
- Feedback map: `amplitude = clip(1.0 - min((alpha - baseline) / (2 * baseline), 1.0), 0.1, 1.0)` where `baseline` is the mean of block 1 — so the subject's own baseline sets the ceiling
- Block 4 runs the same extractors (for logging) but the mapping output goes nowhere
- Include a tutorial block before recording that explains the task without enabling the contingency — NOT counted as an experimental block

**Acceptance:**
1. Full 26-minute session runs to completion
2. HDF5 `/feedback/state` has continuous float32 values during block 2 and NaN during block 4
3. Block transitions produce markers 101–108
4. Offline: reconstructing the feedback signal from raw EEG and matching it to `/feedback/state` agrees within computed-value tolerance (this is the round-trip check that proves the loop was closed correctly)

---

## Phase 9 — File replay + OpenBCI GUI comparison harness

**Goal:** Validation infrastructure. Prove numerical equivalence between our pipeline and OpenBCI GUI on the same input. This is what makes any future result defensible.

**Files to create:**
- `src/openbci_lattice/io/openbci_import.py` — parser for OpenBCI GUI text format (comma-separated, header lines starting with `%`). Returns `(data_uV, timestamps, channel_names, sample_rate)`
- `src/openbci_lattice/validate/replay.py` — runs the pipeline headless (no Qt) on a given input file, producing an HDF5 as if it had been recorded live
- `src/openbci_lattice/validate/compare.py` — takes two HDF5 files and compares: PSD per channel (tolerance 0.1 dB), alpha peak location (tolerance 0.5 Hz), RMS per channel (tolerance 1%)
- CLI entry point `lattice-validate` in `pyproject.toml`
- `tests/test_openbci_import.py` — round-trip parse of a small fixture file

**Files to modify:**
- `src/openbci_lattice/acquisition/board.py` — confirm PLAYBACK_FILE_BOARD works with Cyton+Daisy master_board ID
- `src/openbci_lattice/__main__.py` — `--headless` flag to skip Qt for validation

**Design:**
- OpenBCI GUI's text format: first lines are `%`-prefixed metadata (sample rate, board, etc.), then CSV with sample index, 16 EEG channels in µV, accelerometer, timestamps
- Two validation paths:
  1. **Golden-file:** a small recorded session is bundled; pipeline output must match a reference HDF5 bit-for-bit (up to timestamp equality)
  2. **Synthetic round-trip:** generate synthetic data, write to OpenBCI GUI text format, import through our pipeline, confirm it re-emerges unchanged
- Report format: per-channel table of pass/fail with the actual deviation

**Acceptance:**
1. `lattice-validate <openbci.txt>` on a known-good recording prints PASS for all channels
2. A deliberately corrupted file (e.g. scaling all channels by 2×) produces a FAIL with a diagnostic of the right magnitude
3. Replay runs at ≥100× real-time headless (speeds iteration on encoding experiments)
4. Pytest covers the importer parsing edge cases (truncated files, missing samples)

---

## Phase 10 — A-Gate feature extractors (streaming + offline)

**Goal:** Bring the V1/V2/V3/V4 encodings from the experiment plan into the acquisition pipeline so the QDNU inputs can be generated live and logged alongside EEG.

**Files to create:**
- `src/openbci_lattice/encoding/agate.py` — four classes: `AGateV1`, `AGateV2`, `AGateV3`, `AGateV4`. Each implements `encode(window: np.ndarray) -> np.ndarray` returning `(n_channels, 3)` for `(a, b, c)`. Must match the definitions in the QDNU repo — clone the math, do not re-derive it
- `src/openbci_lattice/ui/widgets/encoding_panel.py` — read-only display of current `(a, b, c)` per channel, updating at 2 Hz
- `tests/test_encoding.py` — reference vectors. Take a set of fixed input windows, hand-compute (or import from the QDNU repo's test suite) expected outputs, assert bit-equivalence within float32

**Files to modify:**
- `src/openbci_lattice/ui/main_window.py` — add encoding tab
- `src/openbci_lattice/io/hdf5_writer.py` — optional `/encoding/{name, params}` dataset so a session can log the running encoding alongside raw EEG (useful for the patient-specific calibration test in Phase 3.2 of the experiment plan)

**Design:**
- V4 is the novel one from the experiment plan: `a = relative_power(beta + gamma)`, `c = relative_power(delta + theta)`, `b = alpha_phase_coherence_across_channels`. For streaming, use a Hilbert-transform-based coherence on a 2 s window; for "coherence across channels" choose one reference channel (Pz) and compute mean coherence to all others
- All four must be importable from the QDNU project without modification — the Lattice repo uses them but does not own them. If the QDNU repo exposes the encoders as a Python package, add it as a dependency; if not, vendor the code with a clear `# VENDORED FROM QDNU` header and a version pin
- Performance target: all four encoders on 2 s × 16 ch window in <5 ms. Benchmark in the test suite and fail if regressed

**Acceptance:**
1. A 40 Hz tone across all channels produces a high `a` on V4
2. All four encoders pass reference-vector tests from the QDNU repo
3. Live feedback panel updates smoothly without UI lag
4. Benchmark: `python -m timeit -s 'from openbci_lattice.encoding.agate import AGateV4; import numpy as np; e=AGateV4(250); w=np.random.randn(16, 500).astype(np.float32)' 'e.encode(w)'` reports <5 ms per call

---

## After Phase 10

With everything above in place, the pipeline is ready for the Phase 1 / 2 / 3 work defined in `eeg_experiment_plan.md` (paradigm data collection, encoding comparison, quantum hardware validation). Everything from there on is research, not infrastructure — but new phases can be added here as they come up:

- Phase 11: LSL egress for multi-device sync (eye tracker, stimulus computer)
- Phase 12: Real-time ICA for ocular artifact subtraction in the biofeedback loop
- Phase 13: AWS pipeline — S3 upload hook on session end, Lambda-triggered MNE analysis, DynamoDB metadata index
- Phase 14: Nested channel-subset analysis (16→8→4) runner for the scaling experiments

Add these as the research demands them, not before.
