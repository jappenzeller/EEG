# QPNN Custom EEG Experimental Plan (Revised)

> Revised 2026-02-15 to match actual hardware: OpenBCI Cyton+Daisy (16-ch, 125 Hz).
> Original plan assumed 18 channels at 256 Hz.

## Objective

Determine whether higher-resolution, self-collected EEG data with controlled paradigms can solve the encoding bottleneck identified in the arXiv paper. The paper established that the A-Gate architecture works (99.3% discrimination on hardware) but the encoding from scalp EEG to PN parameters loses too much information (53.4% AUC vs 62.5% classical). This experiment tests whether the problem is the data quality or the encoding math.

---

## Hardware Specifications

| Spec | Value |
|------|-------|
| Board | OpenBCI Cyton + Daisy |
| Channels | 16 differential |
| Sample rate | 125 Hz (16-ch mode) |
| Resolution | 24-bit (ADS1299) |
| Nyquist limit | 62.5 Hz |
| Electrodes | Gold cup + Ten20 paste |
| Target impedance | <10 kΩ |

### Montage (16 channels)

Dropped P7/P8 in favor of Pz + Cz. Rationale:

- **Pz**: Highest-SNR P300 electrode. The A-Gate processes each channel through 14 gates before ring topology — noisy per-channel extraction propagates through entanglement. One clean Pz outweighs two marginal P7/P8.
- **Cz**: Vertex hub, highest amplitude for motor/cognitive signals, natural midline anchor for ring topology.
- **F7/F8 retained**: Non-negotiable — the paper's 8-channel CHB-MIT baseline used Fp1/Fp2/F3/F4/F7/F8/T7/T8. Dropping them kills Phase 4 replication.
- **P7/P8 sacrificed**: Not in the paper's 8-ch set, posterior temporal captures already-propagated seizure dynamics (not the early E-I shifts the PN model targets), P3/P4 still cover parietal bilateral.

```
Cyton (channels 1-8):  Fp1, Fp2, F3, F4, F7, F8, T7, T8
Daisy (channels 9-16): C3, C4, Cz, Pz, P3, P4, O1, O2
Reference: A1/A2 (linked earlobes)
Ground: AFz (forehead midline)
```

### Ring Topology Channel Ordering

For the A-Gate CNOT chain. Traces a spatial path across the scalp so adjacent channels in the entanglement layer are spatially adjacent (nearest-neighbor assumption):

```
Fp1 → F7 → F3 → C3 → T7 → P3 → O1 → Pz → O2 → P4 → T8 → C4 → F4 → F8 → Fp2 → Cz → (ring)
```

### Phase 4 Scaling Subsets (nested)

| Channels | Positions | Notes |
|----------|-----------|-------|
| 16 | Full ring order | Complete montage |
| 8 | Fp1, Fp2, F3, F4, F7, F8, T7, T8 | Matches paper's CHB-MIT exactly |
| 4 | Fp1, Fp2, F7, F8 | Bilateral frontal pairs |

---

## Phase 0: Pre-Hardware Preparation (Now → Hardware Arrival)

### 0.1 Software Pipeline

Build the full recording-to-prediction pipeline before touching hardware. Every minute spent debugging software with the headset on is wasted.

- **Recording interface**: BrainFlow API via `openbci_eeg.acquisition`. Script connects, streams, saves `.npy` with timestamps and channel labels. Test with synthetic board (`BoardIds.SYNTHETIC_BOARD`).
- **Preprocessing pipeline**: Use parameters appropriate for 125 Hz Nyquist:
  - Bandpass filter: 0.5–50 Hz (FIR)
  - Notch filter: 60 Hz (US mains)
  - Normalization: per-channel z-score within each recording session
  - Artifact rejection: amplitude threshold ±150 µV to flag bad segments
- **Encoding pipeline**: All three encodings (V1/V2/V3) ready to run on arbitrary channel counts. Add V4 candidate (see Phase 2).
- **Evaluation pipeline**: QPNN fidelity classifier + XGBoost baseline, identical splits, AUC/accuracy/F1 output. Should run end-to-end in one command.

### 0.2 Channel Placement Design

16 channels, 10-20 positions. See montage section above.

Document exact placement with photos for reproducibility. Use standard 10-20 measurement from nasion/inion. Mark Cz at 50% nasion-inion and 50% preauricular points; all other positions derive from there.

### 0.3 Baseline Data Collection Protocol

Write the session script in advance. Print it out. Tape it to the wall during recording.

---

## Phase 1: Hardware Validation & Baseline Recording

### 1.1 Hardware Checkout (Day 1)

- Verify all 16 channels show signal
- Check impedances (target <10 kΩ, acceptable <20 kΩ)
- Confirm sample rate = 125 Hz (automatic in 16-ch mode)
- Record 2 minutes of eyes-closed resting state
- Visually inspect: do you see alpha rhythm (8-12 Hz) over O1/O2/Pz with eyes closed? If yes, hardware is working.
- Verify Pz shows clear P300-range activity and Cz shows clean vertex signal
- Run internal test signals (send `0`, `-`, `=` commands) to confirm all channels functional

### 1.2 Controlled State Paradigm (Days 1-3)

Since you can't induce seizures, you need controlled brain state transitions that produce measurable E-I shifts. Three paradigms, increasing in relevance:

**Paradigm A: Eyes Open / Eyes Closed (Simplest)**
- 5 min eyes open → 5 min eyes closed → repeat 3x
- Expected signal: alpha power increase (O1, O2, Pz) during eyes closed
- Why it matters: cleanest, most reproducible state transition in EEG. If the encoding can't detect this, it can't detect anything.

**Paradigm B: Cognitive Load Transitions**
- 5 min rest → 5 min mental arithmetic (serial 7s from 1000) → 5 min rest → repeat 2x
- Expected signal: beta increase (frontal, F3/F4), gamma increase up to ~45 Hz (limited by 62.5 Hz Nyquist), alpha suppression
- Why it matters: frontal E-I shift, closer to the dynamics the PN model is designed to capture

**Paradigm C: Hyperventilation (Clinical Standard)**
- 3 min normal breathing → 3 min hyperventilation → 5 min recovery → repeat 1x
- Expected signal: generalized slowing (increased delta/theta), potential spike-wave if predisposed
- Why it matters: standard clinical EEG activation procedure, produces the most seizure-like EEG changes in healthy subjects
- **Safety**: mild dizziness is normal. Stop if symptoms are concerning. You're your own subject — use judgment.

**Session structure per day:**
1. Setup + impedance check (15 min)
2. Paradigm A: 30 min
3. 5 min break
4. Paradigm B: 25 min
5. 5 min break
6. Paradigm C: 15 min
7. Teardown (5 min)

Total: ~1.5 hours recording per session. Run 3 sessions on different days for variability.

### 1.3 Data Volume Target

Per session: ~90 min × 60 sec × 125 Hz × 16 channels = ~10.8M samples
Across 3 sessions: ~32.4M samples

Segment into 30-second windows (matching CHB-MIT) = ~180 segments per session, ~540 total.

Label segments by paradigm condition and transition periods.

---

## Phase 2: Encoding Experiments

### 2.1 Head-to-Head Encoding Comparison

Run all encodings on the same data, same splits. The paper tested V1/V2/V3. Now add:

**V4 (Multi-Scale Band Power)**
- Instead of collapsing all bands into one (a, b, c) triple, use hierarchical encoding:
  - a = (beta + gamma relative power) — excitatory proxy
  - c = (delta + theta relative power) — inhibitory proxy
  - b = alpha phase coherence across channels — coupling proxy
- Rationale: V2 mapped bands to PN params but the mapping was arbitrary. V4 uses the neurophysiological correspondence directly: fast rhythms = excitation, slow rhythms = inhibition, alpha = thalamocortical coupling.
- **Note**: V4 redefines b from Hilbert instantaneous phase to a coherence metric. This changes the Rz gate interpretation in the A-Gate circuit. Coherence values [0, 1] must be mapped to [0, 2π] — simple linear scaling works, but the circuit was designed around instantaneous phase semantics. Track whether this helps or hurts.

**V5 (Learned Encoding — stretch goal)**
- Train a small autoencoder to compress each channel's 30-sec window into (a, b, c) that maximizes fidelity separation
- This is the "encoding as the bottleneck" test: if a learned mapping dramatically improves separation, the problem was always the hand-crafted encoding, not the circuit

### 2.2 Feature Richness Analysis

The paper identified 24 PN params vs 816 classical features as a key gap. Test whether richer quantum encoding helps:

- **3 params/channel** (current): a, b, c → 48 PN params for 16 channels
- **6 params/channel** (double encoding): run A-Gate twice per channel with different time windows, concatenate
- **9 params/channel** (triple encoding): three time scales (1s, 5s, 30s windows)

Each requires proportionally more qubits. Track whether additional parameters improve AUC or just add noise.

### 2.3 Success Criteria

| Outcome | AUC on Your Data | Interpretation | Next Step |
|---------|-----------------|----------------|-----------|
| Encoding solved | > 0.80 quantum | Clean signal + right encoding fixes the bottleneck | Write follow-up paper, test on CHB-MIT with V4 |
| Partial improvement | 0.65–0.80 quantum | Better data helps but encoding still loses info | Focus on V5 learned encoding |
| No improvement | < 0.65 quantum | Problem is fundamental to 3-param compression | Rethink PN→qubit mapping entirely |
| Classical also fails | < 0.65 classical | Your paradigm doesn't produce separable states | Redesign paradigm, not encoding |

The critical comparison is quantum vs classical on YOUR data with YOUR channels. If classical XGBoost on 16 channels hits 85%+ and quantum is still at 55%, the encoding bottleneck is confirmed on clean data. If both fail, the paradigm is wrong.

---

## Phase 3: Quantum Hardware Validation on Own Data

### 3.1 Run on IBM Hardware

Once you have segments that show separation in simulation:

- Encode best-performing paradigm segments using best encoding
- Run on ibm_torino (or whatever's available)
- Compare simulator vs hardware fidelity distributions
- This closes the loop: own data → own encoding → real hardware

### 3.2 Patient-Specific Calibration Test

Your Julia set analysis revealed boundary crossing direction varies by subject. With your own data across sessions:

- Train template on Session 1, test on Session 2 and 3
- Does the direction stay consistent within-subject across days?
- If yes: patient-specific calibration works. If no: the encoding is session-dependent (worse problem).

---

## Phase 4: Scaling Argument (If Phase 2 Succeeds)

If you achieve > 0.80 AUC with quantum encoding on your 16-channel data:

- Run the same experiment dropping to 8 channels (Fp1, Fp2, F3, F4, F7, F8, T7, T8 — matches paper), then 4 (Fp1, Fp2, F7, F8)
- Plot AUC vs channel count for both quantum and classical
- Show where the curves cross (if they do)
- This is the scaling advantage argument: at what M does O(M) quantum match or beat O(M²) classical?
- Channel subsets are nested, so the 8-ch and 4-ch results are directly comparable to the paper's CHB-MIT baseline

---

## Equipment Checklist

- [ ] Cyton + Daisy board (16-ch)
- [ ] Gold cup electrodes (20 — 16 active + ref + ground + spares)
- [ ] Ten20 conductive paste
- [ ] NuPrep skin prep gel
- [ ] Touch-proof to header pin adapters (if needed)
- [ ] USB dongle (included with Cyton)
- [ ] Measuring tape (nasion-inion for 10-20 placement)
- [ ] Skin-safe marker
- [ ] Mirror or second person for electrode placement
- [ ] Medical tape for electrode securing
- [ ] Timer / metronome app for paradigm pacing
- [ ] Recording laptop with pipeline installed
- [ ] Backup storage for raw data

## Timeline

| Week | Activity |
|------|----------|
| Now–Week 2 | Phase 0: Build pipeline, test with synthetic data |
| Week 2.5 | Hardware arrives, Phase 1.1: checkout |
| Week 3 | Phase 1.2: Record 3 sessions |
| Week 3-4 | Phase 2: Run encoding experiments |
| Week 4-5 | Phase 3: Hardware validation if results warrant |
| Week 5+ | Phase 4 or redesign based on results |

## Key Principles

1. **Pipeline first, hardware second.** Every component should work on synthetic data before the headset arrives.
2. **Paradigm A is the gating test.** If eyes-open/closed alpha doesn't separate, nothing downstream will work. Don't skip to harder paradigms.
3. **Classical baseline on every experiment.** No quantum result means anything without the XGBoost comparison on the same data.
4. **Document everything.** Photos of electrode placement, impedance values, session notes, environmental conditions. This is your data for paper #2.
5. **Single subject is fine for now.** You're testing encoding feasibility, not clinical generalization. N=1 with controlled paradigms is the right scope.
6. **Ring topology ordering matters.** Always feed channels to the A-Gate in the defined spatial ring order. Scrambled ordering breaks the nearest-neighbor correlation assumption.

## Revision History

| Date | Change |
|------|--------|
| 2026-02-15 | Updated from 18-ch/256 Hz to 16-ch/125 Hz to match actual hardware |
| 2026-02-15 | Montage: dropped P7/P8, added Pz/Cz based on A-Gate architecture analysis |
| 2026-02-15 | Defined ring topology channel ordering for CNOT chain |
| 2026-02-15 | Defined nested scaling subsets: 16 → 8 → 4 channels |
| 2026-02-15 | Corrected bandpass upper limit from 128 Hz to 50 Hz (Nyquist = 62.5 Hz) |
| 2026-02-15 | Corrected data volume estimates |
| 2026-02-15 | Added V4 encoding note about b-parameter semantic change |
| 2026-02-15 | Updated equipment checklist to match actual purchase |
