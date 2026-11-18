# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
# Pre-registration: vector-calculus upgrade, membrane-arena-3x3 v0.2.1
#
# Frozen BEFORE the evaluation split was measured. Calibration split:
# seeds 42, 1, 7, configs in `standard_configs()` (stock, offset_ladder,
# exact, planted_collision, high_coupling). Evaluation split: seeds
# 0, 2, 3, 123, 2026, 11, 99 — never used for any threshold choice.
# Thresholds and bars below are never moved to force a pass.

## Definitions

- Scalar field `s[i, j]`: fidelity of cell `(i, j)` to its reference.
- Flow `F = -grad s` on cell faces; closed lattice (zero boundary flux).
- Divergence: finite-volume net outward face flux per cell.
- Escape event at step `t`: the step is rejected with a divergence-band
  guard reason — `pairwise_divergence_below_minimum`,
  `divergence_below_minimum`, or `divergence_above_maximum`.
- Features are computed from the PRE-step state: one-step-ahead prediction.
- Baseline score: stock scalar `max(1 - fidelity)` (`max_infidelity`).
  Secondary baseline: largest per-cell fidelity drop across the previous
  step (`max_fidelity_drop`).

## Frozen thresholds (mirrored as FROZEN_* in vector_calculus.py)

- `FROZEN_DIV_FLAG = 0.15`: flag a step high-divergence when
  `max |div| > 0.15`. Calibration: overall max|div| p50 0.067, p90 0.507;
  escape mean 0.613 vs non-escape mean 0.082.
- `FROZEN_FID_FLAG = 0.60`: flag on the baseline when
  `max(1 - fidelity) > 0.60`. Calibration: escape mean infidelity 0.473
  vs non-escape 0.378.
- `FROZEN_MIN_PRECISION_LIFT = 2.0`, `FROZEN_MIN_RECALL = 0.30`,
  `FROZEN_MIN_AUC_GAIN = 0.05`.

## Claims

- (i) Divergence-as-escape-predictor: PASS iff, at the frozen flag,
  precision >= 2.0x the escape base rate AND recall >= 0.30.
  Calibration: precision 0.667, recall 0.667, lift 3.16x (45 escapes /
  213 steps, base rate 0.211).
- (ii) Discrete Gauss identity holds exactly: PASS iff max |residual|
  over 200 random fields plus every real lattice field in a stock run
  is < 1e-12. Validation tautology (telescoping); reported as such.
- (iii) Discrete Stokes identity holds exactly: PASS iff max |residual|
  < 1e-12 on the same fields. Validation tautology; reported as such.
- (iv) Divergence beats the fidelity-drop baseline: PASS iff
  AUC(max|div|) - AUC(max infidelity) >= 0.05 on the same labelled
  steps. Calibration: AUC div 0.779, AUC fid 0.610, gain 0.169.

## Calibration observations (not claims)

- The 15 false negatives at every flag in 0.08–0.30 are the exact-mode
  escapes: global synchronisation shrinks fidelity differences, so
  divergence is small when the whole lattice exits the band from below.
  Divergence is expected to predict pair-specific escapes (collision /
  flying apart), not global-sync escapes. This asymmetry is part of the
  honest-limits report, not a reason to move any threshold.
- `max_fidelity_drop` AUC on calibration is 0.248 (below chance):
  escapes carry SMALLER prior fidelity drops than non-escapes, because
  planted collisions escape at step 1 with no prior drop and exact-mode
  sync raises fidelity. Reported as a secondary baseline, no claim on it.
- Escapes in this dataset are config-constructed (planted collision,
  offset ladder, exact sync), not spontaneous stock-run events: stock
  and high-coupling runs produced 0 escapes on calibration seeds.
  The predictor is measured on that constructed mix, and the report says so.
