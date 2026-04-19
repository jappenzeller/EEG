# Canonical Wiring

This is the single source of truth for channel -> board -> pin -> wire color -> 10-20 position.
If any other file in this repo disagrees, it is wrong and must be corrected to match this one.

Last verified against physical hardware: 2026-04-19
Montage version: `2026-04-19-midline`

## Channel Map

| Channel | Board | Pin | Wire | Hex | 10-20 Position | Region |
|---|---|---|---|---|---|---|
| 1 | Cyton | N1P | grey | `#9AA4AE` | Fp1 | L frontal pole |
| 2 | Cyton | N2P | purple | `#8A2BE2` | Fp2 | R frontal pole |
| 3 | Cyton | N3P | blue | `#1E90FF` | F3 | L frontal |
| 4 | Cyton | N4P | green | `#2E8B57` | F4 | R frontal |
| 5 | Cyton | N5P | yellow | `#FFD700` | F7 | L lateral frontal |
| 6 | Cyton | N6P | orange | `#FF8C00` | F8 | R lateral frontal |
| 7 | Cyton | N7P | red | `#D9342B` | Fz | frontal midline |
| 8 | Cyton | N8P | brown | `#8B5A2B` | Oz | occipital midline |
| 9 | Daisy | N1P | grey | `#9AA4AE` | C3 | L central |
| 10 | Daisy | N2P | purple | `#8A2BE2` | C4 | R central |
| 11 | Daisy | N3P | blue | `#1E90FF` | Pz | midline parietal |
| 12 | Daisy | N4P | green | `#2E8B57` | Cz | midline central |
| 13 | Daisy | N5P | yellow | `#FFD700` | P3 | L parietal |
| 14 | Daisy | N6P | orange | `#FF8C00` | P4 | R parietal |
| 15 | Daisy | N7P | red | `#D9342B` | O1 | L occipital |
| 16 | Daisy | N8P | brown | `#8B5A2B` | O2 | R occipital |

Full midline spine: Fz -> Cz -> Pz -> Oz.

## Reference and Ground

No Y-splitter. Two independent SRB2 references and two independent BIAS electrodes.

| Role | Position | Pin | Wire |
|---|---|---|---|
| REF (Cyton) | Left earlobe (A1) | Cyton SRB2 | -- |
| REF (Daisy) | Right earlobe (A2) | Daisy SRB2 | -- |
| BIAS (front) | AFz forehead | Cyton BIAS | black |
| BIAS (back) | POz midline posterior | Daisy BIAS | black |

## Positions NOT in this montage

T7, T8 (bilateral temporal -- dropped for midline completion), P7, P8, Oz (reserved: AFz as front BIAS, POz as back BIAS).

## Notes

- The signal pin at each labeled position on both boards is the **top pin adjacent to the silkscreen label** (the N-input row). AGND is a PCB-level pin and is not used for subject connection.
- Channel N on Cyton and channel N+8 on Daisy share the same wire color. This is a design invariant -- the colors cycle identically per board.
- Rainbow is in **reverse** channel order: grey = ch 1/9, brown = ch 8/16. Black is reserved for BIAS.
- Average re-referencing (A1+A2) is done post-acquisition in MNE, not at the hardware level.

## Montage History

### 2026-04-19-verified (previous)
Ch 7 = T7, Ch 8 = T8, Ch 11 = Pz, Ch 12 = Cz, back BIAS at inion.

### 2026-04-19-midline (current)
Ch 7 = Fz, Ch 8 = Oz, back BIAS at POz.
Rationale: drop bilateral temporal (Octabolt fit issues, EMG contaminated, low A-Gate info) in favor of completing the midline spine Fz+Cz+Pz+Oz for FM-theta capture and improved posterior spatial gradient.
