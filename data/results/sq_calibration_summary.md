# Signal Quality Calibration Summary

**Date:** 2026-04-19
**Hardware:** OpenBCI Cyton+Daisy, 16ch @ 125 Hz, COM4
**Reference:** SRB on both earlobes (gel), BIAS not connected
**Test:** Gel finger touch on CH5/F7 (yellow wire) and CH6/F8 (orange wire)

## Data Files

| File | Description |
|------|-------------|
| `data/results/sq_session_gel_finger.json` | Full session: 129 timepoints × 16 channels × 3 metrics (rail%, std, 60Hz ratio) + channel summary + touch events + observations |
| `charts/gel_finger_test/*.png` | 7 visualization charts (heatmaps, status timeline, time series, 3D scatter, channel summary, correlation, Cyton vs Daisy) |

## Threshold Calibration (current values)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `RAIL_THRESHOLD_UV` | 500 uV | DC-corrected (mean subtracted). Catches true ADC clipping without false-flagging DC offset from earlobe SRB |
| `GREEN_MAX_PCT` | 5% | Rail |
| `YELLOW_MAX_PCT` | 50% | Rail |
| `FLATLINE_STD_UV` | 1.0 uV | Dead channel (bias-pinned constant) |
| `LOW_ACTIVITY_STD_UV` | 5.0 uV | Suspicious — too quiet for real EEG |
| `HIGH_ACTIVITY_STD_UV` | 50.0 uV | Above this = noisy reference or floating |
| `SATURATION_STD_UV` | 500.0 uV | Floating electrode territory |
| `LINE_NOISE_GREEN_MAX` | 0.30 | <30% 60Hz power = BIAS connected, good CMR |
| `LINE_NOISE_YELLOW_MAX` | 0.85 | >85% = pure antenna, floating electrode |

## Status Logic

```
status = worst-of(rail_check, std_check, line_noise_check)

GREEN:  rail <5%  AND  5 < std < 50 uV  AND  60Hz < 30%
YELLOW: rail 5-50%  OR  std 1-5 or 50-500 uV  OR  60Hz 30-85%
RED:    rail ≥50%  OR  std <1 or >500 uV  OR  60Hz >85%
```

## Observed Signal Profiles

### By electrode state

| State | Rail % | Std (uV) | 60 Hz | Status | Notes |
|-------|--------|----------|-------|--------|-------|
| Floating (Cyton) | 80-100 | 400-2000 | 85-99% | RED | Antenna picking up mains |
| Floating (Daisy, bias-pinned) | 0-30 | 50-170 | 0% | YELLOW (false) | Quiet DC, no real signal |
| Gel finger, SRB only | 0 | 70-210 | 30-85% | YELLOW | Good contact, noisy reference |
| Gel finger, best contact (F7) | 0 | 48.8 | 0% | GREEN (1 tick) | Borderline, needs BIAS for sustained |
| Gel finger + gel earclip, no BIAS | 0 | 180-210 | 32-81% | YELLOW | Earclip gel didn't help much |
| BIAS touched (bare fingers + gel) | 0 | 0.0 | 0% | RED (flatline) | BIAS feedback loop latched — too low impedance |

### By board

| Board | Median Std | Median 60 Hz | Median Rail | Behavior |
|-------|-----------|-------------|------------|----------|
| Cyton (CH1-8) | 200-530 | 60-74% | 0-47% | High 60Hz antenna effect, responsive to touch |
| Daisy (CH9-16) | 94-330 | 0-57% | 0-6% | Bias-pinned quiet, false yellows on T7/T8/P3/P4 |

## Touch Events Detected

7 sustained contact periods on F7/F8 (≥3 seconds each):

| # | Time Range | Duration | F7 Avg Std | F7 Best Std | F7 Avg 60Hz |
|---|-----------|----------|-----------|------------|-------------|
| 1 | t=1-8 | 7.1s | 253.5 | 171.1 | 10% |
| 2 | t=13-15 | 2.1s | 110.2 | 76.8 | 5% |
| 3 | t=18-54 | 36.3s | 99.3 | 71.7 | 44% |
| 4 | t=57-77 | 20.3s | 71.6 | 48.8 | 18% |
| 5 | t=81-94 | 13.0s | 79.5 | 51.2 | 5% |
| 6 | t=100-115 | 15.0s | 105.4 | 53.7 | 14% |
| 7 | t=118-123 | 5.0s | 188.2 | 112.2 | 14% |

Touch 4 (t=57-77) produced the best result: F7 std=48.8 uV, briefly GREEN.

## Channel Correlation

- **Cyton channels strongly correlated** (shared SRB reference path, all respond to touch together)
- **Daisy channels correlated internally** (shared bias drift)
- **Weak cross-board correlation** — Cyton and Daisy behave independently, suggesting separate analog front-ends

## Key Findings

1. **GREEN requires BIAS.** SRB-only reference caps best-case at ~50 uV std (borderline GREEN). With BIAS driving common-mode cancellation, std should drop to 5-20 uV range.

2. **BIAS cannot be tested with bare fingers** — impedance too low, causes feedback loop latch. Must use proper scalp electrodes (5-10 kOhm).

3. **Daisy false yellows** — T7, T8, P3, P4 show persistent YELLOW despite no contact (bias-pinned at std 50-170, 0% 60Hz). A "suspiciously clean" rule (0% 60Hz + no contact) could catch this.

4. **DC correction was essential** — without subtracting per-channel mean, all channels showed 100% railed due to SRB DC offset.

5. **60Hz ratio oscillates** at ~2s period in the sq-log — this is the 2-second analysis window sliding. Not a real oscillation.

## Next Steps

- [ ] Test with proper scalp electrode placement (SRB + BIAS + gel)
- [ ] Add "suspiciously clean" check for bias-pinned channels (0% 60Hz + std 50-170)
- [ ] Evaluate whether to add 1/f spectral shape check for stronger dead-channel detection
- [ ] Calibrate thresholds on real scalp EEG data
- [ ] Consider per-board threshold adjustment (Cyton vs Daisy behave differently)
