# Results: Crystallize injection analysis (SYNTHETIC)

SYNTHETIC classical NumPy simulation of a toy model. Nothing here is a physical claim or a statement about any real system. Sole author: Jean-François Brisson. Every number below is produced by a committed script under `experiments/` and stored under `experiments/results/`; the seeds are listed in each section. The protocol was committed before the confirmatory run: `docs/PREREGISTRATION-injection.md` (PR #2, merge commit b021b0d); the analysis code is PR #3, the phase damping experiment PR #4, the reconciliation PR #5, the controls PR #6.

Environment: Python 3.12.14 and NumPy 2.5.4 on one Linux machine for all generated outputs (CI runs the tests on Python 3.10 to 3.13). Nothing in this document changes the published model, a guard, a threshold, the breaker or any default.

## 1. Summary

Findings, with the section that supports each:

1. **The injection adds steps in every regime and does not cure the breaker** (section 3). Accepted steps rise by 2.40 (stock), 8.93 (intermediate) and 56.99 (evolved genome) per run, each 95% interval excluding zero, all Holm-significant. All 100 runs per regime still ended with the breaker open (Wilson 95% interval 0.963 to 1.000), the same as the stock engine.
2. **No fixed commit is right in every regime** (section 3). Fixed 0.35 has the lower first-forecast Brier in the stock regime (0.212 against 0.413 for fixed 0.75), fixed 0.75 is lower in the intermediate (0.257 against 0.306) and evolved (0.109 against 0.395) regimes. All three differences are significant after Holm correction. The 0.35 default was chosen on stock seeds 42, 0, 1, 2, 3; on 100 disjoint confirm seeds it still sits close to the stock regime's pass rate (0.317), so the default is not an artefact of those five seeds, but it is a stock-regime default only.
3. **The adaptive commit tracks the regime but pays a burn-in cost** (section 3). Starting from 0.35 it beats fixed 0.35 in the intermediate and evolved regimes (Brier difference -0.036 and -0.287) and loses to it in the stock regime (+0.024). Against a fixed commit tuned in advance on dev seeds (FT) it is worse in all three regimes (differences +0.026, +0.033, +0.024). Its advantage is that it needs no tuning, not that it is more accurate.
4. **The event pass rate is mostly a statement about run length** (section 4, exploratory). Every run ends with a streak of 5 failed events by construction. Before that streak the injection passes 0.938, 0.984 and 0.995 of events (stock, intermediate, evolved); the regime-level rates 0.317, 0.633 and 0.914 are mainly the share of the run that is not the final streak. A first commit that ignores the position in the run cannot be calibrated within a run (section 3, reliability tables).
5. **The ledger score depends on the scoring rule** (section 5). The merged per-attempt rule and the design note's two-forecast rule give different signs in the intermediate regime. This is a definitional decision, not a measurement.
6. **The design note's numbers are reconciled** (section 6). Two differences, the injection RNG stream and the scoring rule, explain every mismatch; with both, all 5 stock rows and the evolved seed-42 row match the note exactly. The note's evolution fitness figures and the held-out seed 1 counts (off by one) are not reproduced.
7. **True phase damping extends runs, up to a point** (section 7, experiment). Off-diagonal damping at the stock rate gives 17.8 steps and 12.8 accepted per run against 8.7 and 3.6 for the published term, paired over 100 seeds. At a rate of 1.0 or more it is worse than the published term. The published dephasing term measurably raises purity and moves populations; true damping lowers purity and moves none. It is a change to the published model and is not adopted.
8. **Controls behave as expected** (section 8). Under fair-coin outcomes no policy scores above zero; fixed policies are unchanged by event order; the adaptive rule changes little under permutation. Its pseudo-count of 4 is not optimal anywhere.

## 2. Design

Seeds: dev 1000 to 1099, confirm 2000 to 2099, tuning 42, 0, 1, 2, 3, pilot 900 to 909, phase damping 3000 to 3099 (attribution 3000 to 3039). Regimes: stock, intermediate (arithmetic midpoint of stock and evolved parameters, fixed by definition) and the evolved genome supplied with the design note, used as given. Policies: F35, F75, A35, A75 and FT (fixed at the dev-set pass rate of the regime). Because the commit never changes acceptance, one run per seed serves every policy and policies are re-scored offline; `tests/test_experiments.py` checks that this equals the engine's own ledger and that the policy cannot change the dynamics. Confirmatory family: 12 two-sided paired sign-flip tests (100,000 flips), Holm corrected; effect sizes with 10,000-resample percentile bootstrap intervals. Full detail is in the protocol.

Deviations from the protocol: none in the analysis. Added without being pre-registered and labelled exploratory: the event layout (section 4), the controls and sensitivity (section 8), the phase damping experiment and the reconciliation. A pytest `pythonpath` setting was added so the experiment helpers can be imported by their tests.

## 3. Confirmatory results (confirm seeds 2000 to 2099)

| regime | dev pass rate (events) | confirm pass rate (events) | FT commit | breaker open, injected (Wilson 95%) | breaker open, stock | mean steps stock | mean steps injected | mean accepted stock | mean accepted injected | runs without events |
|---|---|---|---|---|---|---|---|---|---|---|
| stock | 0.316 (766) | 0.317 (756) | 0.32 | 100/100 (0.963 to 1.000) | 100/100 | 8.6 | 11.2 | 3.6 | 6.0 | 0 |
| intermediate | 0.602 (1320) | 0.633 (1402) | 0.60 | 100/100 (0.963 to 1.000) | 100/100 | 8.8 | 17.9 | 3.8 | 12.7 | 0 |
| evolved | 0.918 (6245) | 0.914 (6146) | 0.92 | 100/100 (0.963 to 1.000) | 100/100 | 7.0 | 64.2 | 1.9 | 58.9 | 0 |

First rejection reasons (analysis set, count of events): stock: {'residual_above_target': 746, 'divergence_above_maximum': 10}; intermediate: {'residual_above_target': 182, 'divergence_above_maximum': 1220}; evolved: {'divergence_above_maximum': 6132, 'residual_above_target': 14}

Confirmatory tests, family of 12. A negative mean difference in a Brier row means the first-named policy is better (lower Brier). Probabilities are Monte Carlo sign-flip p-values with a floor of about 1e-5.

| id | regime | comparison | mean difference [95% CI] | dz [95% CI] | p (sign-flip) | p Holm | reject at 0.05 |
|---|---|---|---|---|---|---|---|
| C1 | stock | accepted steps: injection minus no injection | 2.4000 [2.1500, 2.6500] | 1.88 [1.64, 2.23] | 0.00001 | 0.00012 | yes |
| C4 | stock | Brier F35 minus F75 | -0.2011 [-0.2198, -0.1829] | -2.17 [-2.57, -1.90] | 0.00001 | 0.00012 | yes |
| C7 | stock | Brier A35 minus F35 | 0.0235 [0.0193, 0.0271] | 1.19 [0.76, 2.05] | 0.00001 | 0.00012 | yes |
| C10 | stock | Brier A35 minus FT | 0.0257 [0.0223, 0.0286] | 1.60 [1.10, 2.64] | 0.00001 | 0.00012 | yes |
| C2 | intermediate | accepted steps: injection minus no injection | 8.9300 [8.2700, 9.6000] | 2.61 [2.32, 3.03] | 0.00001 | 0.00012 | yes |
| C5 | intermediate | Brier F35 minus F75 | 0.0495 [0.0331, 0.0648] | 0.61 [0.37, 0.90] | 0.00001 | 0.00012 | yes |
| C8 | intermediate | Brier A35 minus F35 | -0.0356 [-0.0444, -0.0270] | -0.79 [-0.99, -0.62] | 0.00001 | 0.00012 | yes |
| C11 | intermediate | Brier A35 minus FT | 0.0328 [0.0291, 0.0359] | 1.88 [1.30, 3.06] | 0.00001 | 0.00012 | yes |
| C3 | evolved | accepted steps: injection minus no injection | 56.9900 [53.8498, 60.2002] | 3.49 [3.16, 3.99] | 0.00001 | 0.00012 | yes |
| C6 | evolved | Brier F35 minus F75 | 0.2862 [0.2814, 0.2907] | 12.09 [10.21, 15.36] | 0.00001 | 0.00012 | yes |
| C9 | evolved | Brier A35 minus F35 | -0.2867 [-0.2936, -0.2796] | -8.05 [-9.85, -6.93] | 0.00001 | 0.00012 | yes |
| C12 | evolved | Brier A35 minus FT | 0.0242 [0.0232, 0.0251] | 5.06 [4.34, 6.07] | 0.00001 | 0.00012 | yes |

Prior expectations recorded in the protocol: all were met in direction. The one that needs care is C5 (intermediate, F35 minus F75): the protocol said small, and the effect is moderate (dz 0.61).

Mean first-forecast Brier per run (lower is better; a forecast of 0.5 always scores 0.25):

| regime | F35 | F75 | A35 | A75 | FT |
|---|---|---|---|---|---|
| stock | 0.2121 | 0.4132 | 0.2356 | 0.3147 | 0.2099 |
| intermediate | 0.3060 | 0.2566 | 0.2704 | 0.2545 | 0.2376 |
| evolved | 0.3948 | 0.1086 | 0.1081 | 0.0865 | 0.0839 |

Pooled expected calibration error, 5 equal-width bins:

| regime | F35 | F75 | A35 | A75 | FT |
|---|---|---|---|---|---|
| stock | 0.033 [0.011, 0.055] | 0.433 [0.411, 0.455] | 0.152 [0.138, 0.173] | 0.375 [0.364, 0.387] | 0.003 [0.000, 0.026] |
| intermediate | 0.283 [0.264, 0.301] | 0.117 [0.099, 0.136] | 0.169 [0.158, 0.179] | 0.202 [0.192, 0.213] | 0.033 [0.014, 0.051] |
| evolved | 0.564 [0.558, 0.569] | 0.164 [0.158, 0.169] | 0.067 [0.062, 0.073] | 0.044 [0.040, 0.048] | 0.006 [0.001, 0.012] |

Reliability of the adaptive rule from 0.35 (bin mean forecast against observed pass rate; the 0.0-0.2 and 0.8-1.0 bins in some regimes are empty or small):

stock:

| bin | events | mean forecast | observed pass rate |
|---|---|---|---|
| 0.0-0.2 | 8 | 0.188 | 0.000 |
| 0.2-0.4 | 269 | 0.342 | 0.368 |
| 0.4-0.6 | 400 | 0.492 | 0.300 |
| 0.6-0.8 | 79 | 0.644 | 0.266 |
| 0.8-1.0 | 0 | n/a | n/a |

intermediate:

| bin | events | mean forecast | observed pass rate |
|---|---|---|---|
| 0.0-0.2 | 0 | n/a | n/a |
| 0.2-0.4 | 103 | 0.351 | 0.971 |
| 0.4-0.6 | 308 | 0.526 | 0.656 |
| 0.6-0.8 | 800 | 0.701 | 0.578 |
| 0.8-1.0 | 191 | 0.825 | 0.649 |

evolved:

| bin | events | mean forecast | observed pass rate |
|---|---|---|---|
| 0.0-0.2 | 0 | n/a | n/a |
| 0.2-0.4 | 100 | 0.350 | 1.000 |
| 0.4-0.6 | 200 | 0.523 | 1.000 |
| 0.6-0.8 | 603 | 0.717 | 0.995 |
| 0.8-1.0 | 5243 | 0.916 | 0.900 |

Reading: in the stock regime the adaptive forecasts of 0.4 to 0.8 are too high (observed 0.27 to 0.30) and in the intermediate and evolved regimes the early forecasts near 0.35 are far too low (observed 0.97 to 1.00), because the first events of a run almost always pass (section 4). The adaptive rule follows the run-level rate and cannot follow the run's own position.

In-sample against held-out. Fixed 0.35 was chosen on the tuning seeds. On those five seeds, under the design note's scoring, its total is +68.0 and fixed 0.75's is -1215.0 (tuning table, section 5). On the 100 disjoint confirm seeds the stock-regime pass rate is 0.317 (dev 0.316, tuning 0.31) and F35's mean first-forecast Brier, 0.212, is within 0.003 of the dev-tuned FT (0.210): the tuning did not overfit those seeds. Its two-forecast ledger on confirm seeds is +12.1 per run with a 95% interval of -4.1 to 28.0, so out of sample the positive sign is weak.

## 4. Event layout inside a run (exploratory, confirm seeds)

| regime | mean events per run | mean terminal failure streak | pass rate, all events | pass rate, events before the terminal streak |
|---|---|---|---|---|
| stock | 7.6 | 5.00 | 0.317 | 0.938 |
| intermediate | 14.0 | 5.00 | 0.633 | 0.984 |
| evolved | 61.5 | 5.00 | 0.914 | 0.995 |

Pass rate by event index within a run (all events, 0-based index):

| regime | event 0 | events 1-2 | events 3-5 | events 6-11 | events 12+ |
|---|---|---|---|---|---|
| stock | 96/100 = 0.96 | 115/200 = 0.57 | 26/296 = 0.09 | 3/160 = 0.02 | n/a |
| intermediate | 100/100 = 1.00 | 199/200 = 0.99 | 272/300 = 0.91 | 270/551 = 0.49 | 47/251 = 0.19 |
| evolved | 100/100 = 1.00 | 200/200 = 1.00 | 300/300 = 1.00 | 600/600 = 1.00 | 4417/4946 = 0.89 |

An event passes when any of up to 3 shrunk variants passes the unmodified guards. Early events almost always pass; after the lattice has drifted too far no variant passes, and 5 consecutive failures latch the breaker. The regime pass rates are therefore close to (events - 5) / events and mostly reflect run length. This also means that "pass rate" is not a stationary quantity within a run.

## 5. Ledger scores under both scoring rules (exploratory)

The merged engine scores every variant tried (cost not subtracted). The design note scores each event twice, once at the first and once at the last commit, averages the two, and subtracts 0.8 per variant. Mean per run on the confirm seeds, with 95% bootstrap intervals:

Per-attempt scoring is the merged engine's; two-forecast scoring is the design note's (cost of 0.8 per variant subtracted).

| regime | scoring | F35 | F75 | A35 | A75 | FT |
|---|---|---|---|---|---|---|
| stock | per-attempt | 434.4 [412.9, 456.3] | -1103.2 [-1145.0, -1064.9] | 91.9 [29.5, 154.7] | -696.0 [-750.9, -642.8] | 500.8 [475.0, 527.1] |
| stock | two-forecast | 12.1 [-4.1, 28.0] | -249.0 [-267.1, -230.7] | -16.9 [-33.4, 0.8] | -78.4 [-85.1, -71.0] | 7.5 [-12.3, 27.0] |
| intermediate | per-attempt | 44.8 [-7.5, 95.7] | -718.2 [-785.7, -649.5] | -702.1 [-755.7, -645.7] | -1003.0 [-1067.2, -937.6] | -190.8 [-220.2, -160.8] |
| intermediate | two-forecast | -429.2 [-476.4, -383.2] | 221.7 [171.7, 272.6] | 30.4 [-0.0, 64.2] | 274.6 [227.3, 324.5] | 141.4 [117.8, 165.5] |
| evolved | per-attempt | -3213.4 [-3448.0, -2984.4] | 2813.9 [2556.6, 3080.9] | 2315.0 [1990.6, 2653.4] | 2720.4 [2380.1, 3073.2] | 2912.6 [2563.2, 3273.6] |
| evolved | two-forecast | -3727.5 [-3958.6, -3501.9] | 3720.4 [3479.0, 3965.7] | 4054.8 [3743.2, 4372.0] | 4605.6 [4288.9, 4927.6] | 4624.7 [4309.9, 4946.1] |

In-sample totals on the tuning seeds (sum over the 5 runs):

| regime | scoring | F35 | F75 | A35 | A75 | FT |
|---|---|---|---|---|---|---|
| stock | per_attempt | 1966.1 | -5071.9 | 434.5 | -3092.7 | 2262.4 |
| stock | two_forecast | 68.0 | -1215.0 | -74.2 | -376.0 | 47.2 |
| intermediate | per_attempt | 277.5 | -3762.5 | -3265.5 | -4593.5 | -1010.0 |
| intermediate | two_forecast | -2086.6 | 933.4 | 58.3 | 1268.0 | 644.6 |
| evolved | per_attempt | -14104.9 | 11753.1 | 8716.3 | 10603.7 | 11352.2 |
| evolved | two_forecast | -16781.8 | 16551.2 | 17623.3 | 20314.1 | 20380.9 |

The two rules disagree in the intermediate regime: fixed 0.35 scores +44.8 per run under the per-attempt rule and -429.2 under the two-forecast rule, and fixed 0.75 the opposite. Under the per-attempt rule the truth of an attempt is "this variant passes", and revised commits are moved toward 0.5 after each miss; under the two-forecast rule the truth is "the event succeeded". A fixed first commit cannot be right for both. This needs a decision by the maintainer; this analysis used the first-forecast Brier, which is rule independent, for all confirmatory claims.

## 6. Reconciliation with the design note

### Note table, commit 0.75, stock parameters

Each cell is `steps / accepted, passes, score`. The last column is the note's printed row.

| seed | merged RNG, merged score | merged RNG, note score | note RNG, merged score | note RNG, note score | note |
|---|---|---|---|---|---|
| 42 | 11 / 6, 3, -970.3 | 11 / 6, 3, -184.7 | 11 / 6, 3, -970.3 | 11 / 6, 3, -184.7 | 11 / 6, 3, -184.7 |
| 0 | 10 / 5, 1, -1120.3 | 10 / 5, 1, -333.1 | 10 / 5, 1, -1120.3 | 10 / 5, 1, -333.1 | 10 / 5, 1, -333.1 |
| 1 | 9 / 4, 1, -964.1 | 9 / 4, 1, -316.7 | 10 / 5, 2, -1121.9 | 10 / 5, 2, -269.9 | 10 / 5, 2, -269.9 |
| 2 | 12 / 7, 4, -1128.1 | 12 / 7, 4, -137.9 | 11 / 6, 3, -970.3 | 11 / 6, 3, -184.7 | 11 / 6, 3, -184.7 |
| 3 | 11 / 6, 2, -889.1 | 11 / 6, 2, -242.5 | 11 / 6, 2, -889.1 | 11 / 6, 2, -242.5 | 11 / 6, 2, -242.5 |

Rows matching the note exactly (steps, accepted, passes, score to one decimal), out of 5:

- merged RNG with merged per-attempt scoring: 0 of 5 (total score -5071.9)
- merged RNG with note two-forecast scoring: 3 of 5 (total score -1215.0)
- note RNG with merged per-attempt scoring: 0 of 5 (total score -5071.9)
- note RNG with note two-forecast scoring: 5 of 5 (total score -1215.0)

### Note totals over the five seeds (two-forecast scoring)

| commit | note total | merged RNG | note RNG |
|---|---|---|---|
| 0.75 | -1215 | -1215.0 | -1215.0 |
| 0.35 | 68 | 68.0 | 68.0 |

### Evolved genome, seed 42 (note: 55 events, 50 passes, fixed 0.35 = -3288.2, adaptive from 0.35 = +3459.2, final commit 0.87)

| RNG | scoring | events | passes | fixed 0.35 | adaptive from 0.35 | adaptive final commit | steps / accepted |
|---|---|---|---|---|---|---|---|
| merged | per-attempt | 56 | 51 | -2920.9 | 2016.3 | 0.873 | 59 / 54 |
| merged | two-forecast | 56 | 51 | -3383.1 | 3585.6 | 0.873 | 59 / 54 |
| note | per-attempt | 55 | 50 | -2719.7 | 1495.9 | 0.871 | 58 / 53 |
| note | two-forecast | 55 | 50 | -3288.2 | 3459.2 | 0.871 | 58 / 53 |

Held-out seed 1 (note: 50 accepted of 55 executed):

| RNG | steps / accepted |
|---|---|
| merged | 56 / 51 |
| note | 56 / 51 |

### Is the RNG stream choice immaterial? 40 dev seeds (1000 to 1039), both streams, per regime

| regime | stream | mean steps | mean accepted | pass rate |
|---|---|---|---|---|
| stock | merged | 11.3 | 6.0 | 0.342 |
| stock | note | 11.2 | 6.0 | 0.340 |
| intermediate | merged | 17.1 | 11.8 | 0.595 |
| intermediate | note | 17.1 | 11.7 | 0.592 |
| evolved | merged | 66.0 | 60.9 | 0.919 |
| evolved | note | 65.6 | 60.5 | 0.919 |

What differs, exactly:

1. **RNG stream.** The merged engine seeds the injection generator with `[seed, 0xC4157A11]`; the note's code used `seed + 10_000_007`. Both draw one uniform per attempt, so the distributions are the same, but the jitter values differ, and seeds 1 and 2 take different paths (steps 9/4 and 12/7 against 10/5 and 11/6 in the note). With the note's stream those rows match.
2. **Scoring.** The merged ledger sums points over every variant tried. The note's ledger averages two forecasts per event and subtracts 0.8 per variant. With the note's scoring the totals are exactly -1215.0 (commit 0.75) and +68.0 (commit 0.35) under either stream, and the per-seed scores match for the seeds whose path agrees.
3. With both differences, 5 of 5 stock rows and the evolved genome seed-42 row (55 events, 50 passes, -3288.2 fixed, +3459.2 adaptive, final commit 0.87) match the note exactly.
4. The stream choice does not matter statistically: over 40 dev seeds per regime the two streams give the same mean steps, accepted steps and pass rates to within 0.5 steps.

Not determined: the note's held-out seed 1 count (50 accepted of 55 executed; here 51 of 56 under both streams) and its averages for seeds 42 and 0 (66.5 and 71.5; here 67.5 and 72.5 under the merged stream), each off by one. A counting convention in the note's evolution harness is the likely reason; that harness was not supplied. The note's evolution fitness values (95.33 and 70.98) and the evolution search itself are not reproduced.

## 7. True phase damping and term attribution (experiment, seeds 3000 to 3099)

The experiment swaps only the dissipative term of a subclassed lattice for off-diagonal damping exp(-gamma |i - j|), a Schur product with a positive semidefinite kernel that keeps trace and diagonal. Guards, thresholds, bounds and the breaker are the stock ones, and both arms start from identical states and Hamiltonians for a given seed.

Paired by seed: both arms start from identical states and Hamiltonians. Difference = true damping minus published.

| gamma | arm | mean steps [95% CI] | mean accepted [95% CI] | breaker open | latch reasons | mean final min purity |
|---|---|---|---|---|---|---|
| 0.0 | published | 6.4 [6.3, 6.5] | 1.4 [1.3, 1.5] | 100/100 | {'divergence_above_maximum': 100} | 0.070 |
| 0.0 | true_phase_damping | 6.4 [6.3, 6.5] | 1.4 [1.3, 1.5] | 100/100 | {'divergence_above_maximum': 100} | 0.070 |
| 0.1 | published | 6.7 [6.6, 6.8] | 1.7 [1.6, 1.8] | 100/100 | {'divergence_above_maximum': 100} | 0.073 |
| 0.1 | true_phase_damping | 7.8 [7.7, 7.9] | 2.8 [2.7, 2.9] | 100/100 | {'divergence_above_maximum': 100} | 0.075 |
| 0.25 | published | 7.9 [7.8, 8.0] | 2.9 [2.8, 3.0] | 100/100 | {'divergence_above_maximum': 100} | 0.093 |
| 0.25 | true_phase_damping | 11.4 [11.3, 11.5] | 6.4 [6.3, 6.5] | 100/100 | {'divergence_above_maximum': 100} | 0.082 |
| 0.5 | published | 8.7 [8.6, 8.8] | 3.6 [3.5, 3.8] | 100/100 | {'residual_above_target': 94, 'divergence_above_maximum': 6} | 0.141 |
| 0.5 | true_phase_damping | 17.8 [17.7, 18.0] | 12.8 [12.7, 13.0] | 100/100 | {'divergence_above_maximum': 100} | 0.089 |
| 1.0 | published | 6.6 [6.5, 6.7] | 1.6 [1.5, 1.7] | 100/100 | {'residual_above_target': 100} | 0.114 |
| 1.0 | true_phase_damping | 6.0 [6.0, 6.1] | 1.0 [1.0, 1.1] | 100/100 | {'purity_too_low': 100} | 0.051 |
| 2.0 | published | 5.8 [5.7, 5.9] | 0.8 [0.7, 0.9] | 100/100 | {'residual_above_target': 100} | 0.108 |
| 2.0 | true_phase_damping | 5.0 [5.0, 5.0] | 0.0 [0.0, 0.0] | 100/100 | {'purity_too_low': 100} | 0.060 |

Paired difference (true damping minus published), accepted steps:

| gamma | mean difference [95% CI] | dz |
|---|---|---|
| 0.0 | 0.00 [0.00, 0.00] | n/a |
| 0.1 | 1.08 [0.99, 1.17] | 2.22 |
| 0.25 | 3.51 [3.40, 3.62] | 6.08 |
| 0.5 | 9.19 [9.01, 9.36] | 10.14 |
| 1.0 | -0.56 [-0.66, -0.46] | -1.12 |
| 2.0 | -0.80 [-0.87, -0.72] | -1.99 |

Purifier check, measured:

Mean change per cell after one application (95% bootstrap CI over 100 seeds). Positive purity change and negative entropy change mean the state became purer. Population shift is the L1 change of the diagonal.

| arm | change in purity | change in normalized entropy | population shift (L1) |
|---|---|---|---|
| published | 0.00637 [0.00630, 0.00644] | -0.01942 [-0.01958, -0.01924] | 0.38589 [0.38542, 0.38636] |
| true_phase_damping | -0.01162 [-0.01165, -0.01158] | 0.05865 [0.05853, 0.05877] | 0.00000 [0.00000, 0.00000] |

Attribution of residual growth by term (method in the table text):

Method (see `attribute_step` in experiments/phase_damping.py): at each step up to and including the first rejection, R is the largest residual over the nine cells. `alone` = R of the candidate built from that term only (then the stock projection and mirror) minus R now; `null_mirror_only` is the same with no term at all (projection and mirror on the unchanged state) and `alone_net` = alone minus null; `leave-out` = R(full candidate) minus R(candidate without the term); the mirror effect = R(full) minus R(full with the mirror off). Per-seed means over steps, then mean and 95% bootstrap CI over seeds. Terms interact through projection and mirror, so alone and leave-out need not add up. Units: residual (infinity norm).

| quantity | all steps: mean [95% CI] | step 1 only: mean [95% CI] |
|---|---|---|
| growth_full | 0.04839 [0.04656, 0.05031] | 0.04241 [0.04123, 0.04369] |
| null_mirror_only | 0.00920 [0.00878, 0.00965] | 0.00173 [0.00164, 0.00182] |
| alone_unitary | -0.00378 [-0.00548, -0.00205] | 0.00786 [0.00758, 0.00816] |
| alone_dephasing | 0.06445 [0.06264, 0.06643] | 0.04288 [0.04169, 0.04423] |
| alone_interaction | 0.00920 [0.00878, 0.00965] | 0.00173 [0.00165, 0.00183] |
| alone_calibrator | 0.00902 [0.00860, 0.00946] | 0.00173 [0.00164, 0.00182] |
| alone_net_unitary | -0.01298 [-0.01494, -0.01102] | 0.00613 [0.00584, 0.00645] |
| alone_net_dephasing | 0.05525 [0.05377, 0.05683] | 0.04116 [0.03996, 0.04251] |
| alone_net_interaction | 0.00001 [0.00000, 0.00001] | 0.00001 [0.00001, 0.00001] |
| alone_net_calibrator | -0.00018 [-0.00018, -0.00017] | 0.00000 [0.00000, 0.00000] |
| leave_out_unitary | -0.01591 [-0.01791, -0.01400] | -0.00047 [-0.00119, 0.00028] |
| leave_out_dephasing | 0.05230 [0.05083, 0.05388] | 0.03455 [0.03335, 0.03586] |
| leave_out_interaction | 0.00000 [0.00000, 0.00000] | -0.00000 [-0.00000, 0.00000] |
| leave_out_calibrator | -0.00009 [-0.00009, -0.00008] | 0.00000 [0.00000, 0.00000] |
| mirror_effect | 0.01242 [0.01190, 0.01297] | 0.00448 [0.00419, 0.00479] |
| dpurity_alone_unitary | 0.01358 [0.01322, 0.01394] | 0.00766 [0.00761, 0.00771] |
| dpurity_alone_dephasing | 0.03029 [0.02970, 0.03088] | 0.01177 [0.01163, 0.01191] |
| dpurity_alone_interaction | 0.00965 [0.00932, 0.00997] | 0.00423 [0.00420, 0.00426] |
| dpurity_alone_calibrator | 0.00943 [0.00911, 0.00974] | 0.00421 [0.00418, 0.00424] |
| dpurity_mirror | 0.01193 [0.01158, 0.01226] | 0.00519 [0.00514, 0.00523] |

Reading:
- At the stock rate (gamma 0.5) true damping reaches 17.8 steps and 12.8 accepted steps on average against 8.7 and 3.6 for the published term; every run still latches the breaker, on the divergence guard. This reproduces the note's 17 to 19 steps and 12 to 14 accepted. It is a better regime, not a cure.
- There is an optimum: at gamma 1.0 and 2.0 true damping is worse than the published term and latches on the purity floor.
- The published dephasing term raises purity (+0.0064 per cell for one application), lowers normalized entropy and shifts populations (L1 0.386). That is a purifier; true damping lowers purity (-0.0116) and shifts no population. This is measured on one application at the stock rate on initial cells.
- Residual growth: the dephasing term accounts for nearly all of it at step 1 (0.0412 net of the unchanged-state baseline, against 0.0424 for the full step). The mirror adds 0.0045 at step 1 and 0.0124 averaged over steps; the unitary (commutator) term alone is 0.0061 at step 1 and negative on average; interaction and calibrator are about zero net (1e-5 or below). The note's claim that the mirror is second is only partly supported: at step 1 the unitary term alone is comparable to the mirror.
- Terms interact through the projection and the mirror, so the leave-one-out and alone measures do not add up; both are shown.

## 8. Negative controls and sensitivity (exploratory, confirm seeds)

Confirm seeds, 100 runs per regime. Means per run with 95% bootstrap CIs over runs.

### Control 1: fair coin outcomes (the ledger must not reward any policy)

Mean first-forecast points per event (cost excluded; expected value for fixed p is -400 (p - 0.5)^2, i.e. F35 -9, F75 -25, FT near 0), and mean first-forecast Brier (expected 0.25 + (p - 0.5)^2).

| regime | metric | F35 | F75 | A35 | A75 | FT | fixed 0.5 |
|---|---|---|---|---|---|---|---|
| stock | points | -7.909 [-12.221, -3.613] | -26.818 [-33.979, -19.632] | -8.287 [-11.262, -5.052] | -16.326 [-19.970, -12.385] | -11.651 [-16.825, -6.495] | 0.000 [0.000, 0.000] |
| stock | Brier | 0.270 [0.259, 0.281] | 0.317 [0.299, 0.335] | 0.271 [0.263, 0.278] | 0.291 [0.281, 0.300] | 0.279 [0.266, 0.292] | 0.250 [0.250, 0.250] |
| intermediate | points | -8.750 [-11.588, -5.818] | -25.416 [-30.304, -20.686] | -8.511 [-10.269, -6.476] | -13.209 [-15.146, -11.108] | -4.166 [-6.122, -2.275] | 0.000 [0.000, 0.000] |
| intermediate | Brier | 0.272 [0.265, 0.279] | 0.314 [0.302, 0.326] | 0.271 [0.266, 0.276] | 0.283 [0.278, 0.288] | 0.260 [0.256, 0.265] | 0.250 [0.250, 0.250] |
| evolved | points | -8.457 [-10.193, -6.757] | -25.904 [-28.738, -23.011] | -3.317 [-3.867, -2.730] | -4.669 [-5.280, -4.002] | -72.079 [-76.841, -67.219] | 0.000 [0.000, 0.000] |
| evolved | Brier | 0.271 [0.267, 0.275] | 0.315 [0.308, 0.322] | 0.258 [0.257, 0.260] | 0.262 [0.260, 0.263] | 0.430 [0.418, 0.442] | 0.250 [0.250, 0.250] |

### Control 2: events permuted within each run

Mean first-forecast Brier per run, real order versus permuted order. Fixed policies must be identical (pipeline check); the adaptive rule may change.

| regime | policy | real order | permuted | difference (permuted - real) |
|---|---|---|---|---|
| stock | F35 | 0.212 [0.205, 0.219] | 0.212 [0.205, 0.219] | -0.0000 |
| stock | F75 | 0.413 [0.402, 0.425] | 0.413 [0.402, 0.425] | 0.0000 |
| stock | A35 | 0.236 [0.225, 0.245] | 0.232 [0.221, 0.241] | -0.0038 |
| stock | A75 | 0.315 [0.313, 0.317] | 0.320 [0.318, 0.322] | 0.0058 |
| stock | FT | 0.210 [0.202, 0.218] | 0.210 [0.202, 0.218] | 0.0000 |
| intermediate | F35 | 0.306 [0.300, 0.312] | 0.306 [0.300, 0.312] | 0.0000 |
| intermediate | F75 | 0.257 [0.247, 0.267] | 0.257 [0.247, 0.267] | 0.0000 |
| intermediate | A35 | 0.270 [0.267, 0.274] | 0.272 [0.268, 0.276] | 0.0018 |
| intermediate | A75 | 0.254 [0.248, 0.261] | 0.263 [0.256, 0.269] | 0.0082 |
| intermediate | FT | 0.238 [0.234, 0.242] | 0.238 [0.234, 0.242] | 0.0000 |
| evolved | F35 | 0.395 [0.393, 0.397] | 0.395 [0.393, 0.397] | 0.0000 |
| evolved | F75 | 0.109 [0.106, 0.112] | 0.109 [0.106, 0.112] | 0.0000 |
| evolved | A35 | 0.108 [0.103, 0.114] | 0.110 [0.104, 0.115] | 0.0016 |
| evolved | A75 | 0.087 [0.082, 0.091] | 0.089 [0.084, 0.094] | 0.0022 |
| evolved | FT | 0.084 [0.079, 0.089] | 0.084 [0.079, 0.089] | 0.0000 |

### Sensitivity: pseudo-count of the adaptive rule

Mean first-forecast Brier per run, pooled ECE (5 bins) and mean two-forecast ledger per run. Weight 0 uses the first commit only until the first event has been seen.

| regime | start | weight | Brier | ECE | ledger per run, per-attempt | ledger per run, two-forecast |
|---|---|---|---|---|---|---|
| stock | 0.35 | 0 | 0.320 [0.306, 0.330] | 0.385 [0.361, 0.404] | -649.5 | -125.2 |
| stock | 0.35 | 1 | 0.268 [0.256, 0.278] | 0.265 [0.244, 0.284] | -244.0 | -54.5 |
| stock | 0.35 | 4 | 0.236 [0.225, 0.245] | 0.152 [0.139, 0.173] | 91.9 | -16.9 |
| stock | 0.35 | 16 | 0.219 [0.211, 0.227] | 0.071 [0.061, 0.088] | 315.2 | 3.0 |
| stock | 0.75 | 0 | 0.277 [0.268, 0.283] | 0.352 [0.344, 0.358] | -595.1 | -1.4 |
| stock | 0.75 | 1 | 0.284 [0.277, 0.289] | 0.356 [0.350, 0.361] | -593.2 | -20.6 |
| stock | 0.75 | 4 | 0.315 [0.313, 0.317] | 0.375 [0.363, 0.387] | -696.0 | -78.4 |
| stock | 0.75 | 16 | 0.366 [0.359, 0.373] | 0.405 [0.387, 0.425] | -900.2 | -166.8 |
| intermediate | 0.35 | 0 | 0.284 [0.277, 0.291] | 0.334 [0.322, 0.347] | -1302.4 | 148.3 |
| intermediate | 0.35 | 1 | 0.273 [0.267, 0.278] | 0.243 [0.233, 0.253] | -1062.7 | 139.2 |
| intermediate | 0.35 | 4 | 0.270 [0.267, 0.274] | 0.169 [0.157, 0.178] | -702.1 | 30.4 |
| intermediate | 0.35 | 16 | 0.280 [0.277, 0.282] | 0.161 [0.145, 0.177] | -274.7 | -168.7 |
| intermediate | 0.75 | 0 | 0.257 [0.251, 0.262] | 0.271 [0.261, 0.280] | -1163.2 | 291.6 |
| intermediate | 0.75 | 1 | 0.256 [0.250, 0.262] | 0.244 [0.235, 0.253] | -1114.8 | 288.3 |
| intermediate | 0.75 | 4 | 0.254 [0.248, 0.261] | 0.202 [0.191, 0.213] | -1003.0 | 274.6 |
| intermediate | 0.75 | 16 | 0.255 [0.247, 0.263] | 0.156 [0.143, 0.170] | -858.1 | 250.0 |
| evolved | 0.35 | 0 | 0.092 [0.087, 0.097] | 0.084 [0.079, 0.090] | 2566.2 | 4512.3 |
| evolved | 0.35 | 1 | 0.094 [0.089, 0.100] | 0.072 [0.067, 0.077] | 2525.5 | 4430.0 |
| evolved | 0.35 | 4 | 0.108 [0.103, 0.114] | 0.067 [0.062, 0.073] | 2315.0 | 4054.8 |
| evolved | 0.35 | 16 | 0.153 [0.148, 0.159] | 0.184 [0.180, 0.196] | 1566.4 | 2822.5 |
| evolved | 0.75 | 0 | 0.085 [0.081, 0.090] | 0.077 [0.073, 0.083] | 2710.2 | 4656.3 |
| evolved | 0.75 | 1 | 0.085 [0.081, 0.090] | 0.070 [0.065, 0.075] | 2714.4 | 4648.8 |
| evolved | 0.75 | 4 | 0.087 [0.082, 0.091] | 0.044 [0.040, 0.048] | 2720.4 | 4605.6 |
| evolved | 0.75 | 16 | 0.090 [0.085, 0.094] | 0.020 [0.017, 0.022] | 2761.0 | 4466.2 |

Reading:
- Fair-coin outcomes: fixed 0.5 scores exactly 0 and every other policy is negative, as the expected value -400 (p - 0.5)^2 requires. The ledger does not reward a forecast it cannot support.
- Within-run permutation leaves every fixed policy's Brier unchanged to 1e-12 (asserted by the script) and moves the adaptive rule's by at most 0.008, so the adaptive gains reported above come from tracking the regime's rate, not from exploiting event order.
- Pseudo-count: a count of 4 is not optimal. From a start far from the regime's rate a small count is better (evolved from 0.35: Brier 0.092 at 0, 0.108 at 4, 0.153 at 16); from a start near the rate a larger count is better (stock from 0.35: 0.320 at 0, 0.236 at 4, 0.219 at 16). A count of 0 makes the commit jump to 0.01 or 0.99 after one event and is the worst choice in the stock regime from 0.35. The value 4 is a compromise with no tuning behind it.
- Guards: `tests/test_injection_guards.py` checks that every committed state passes a fresh observer and is a valid density matrix, that rejected steps leave the state bit-identical, that observer, calibrator, failure limit and lattice settings equal the stock ones after an adaptive run, and that every accepted step corresponds to one stock-checked candidate.

## 9. Limitations

- One toy model and three synthetic parameter regimes; the evolved genome came from the design note and was not re-evolved here, and the intermediate regime is a midpoint, not a fitted one.
- Forecasts use no event features, so the ledger measures calibration to a base rate, not discrimination. The event layout shows a feature that would matter (position in the run) that no policy here uses.
- p-values are Monte Carlo with a floor near 1e-5, and with 100 seeds and deterministic dynamics the standardised effects are very large; the intervals, not the p-values, carry the magnitude.
- All generated numbers are from one machine, Python 3.12.14 and NumPy 2.5.4. Other platforms were not run for these outputs.
- The per-attempt ledger and the two-forecast ledger disagree in sign in one regime (section 5); conclusions about "scores positive" depend on the rule.
- The phase damping sweep varies only the damping rate in the stock engine without injection; it was not combined with the injection, and the attribution covers the first rejection only.
- The design note's evolution results, its non-neighbour fidelity and return-time measurements, and its fitness values were not reproduced.

## 10. Decisions for the maintainer

1. Which ledger definition should the package use (merged per-attempt, or the note's two-forecast)? Changing it alters ledger values only, not dynamics.
2. Whether to adopt the note's injection RNG stream (`seed + 10_000_007`) for bit-identical agreement with the note; it changes which paths seeds 1 and 2 take but not the statistics.
3. Whether a default of 0.35 should stay, be documented as stock-regime only (as now), or be replaced by the adaptive commit; no data here favours changing a default.
4. Whether true phase damping should ever enter the model. It is a model change and is left as an experiment.
5. Any release or archive is a separate decision; none was made.
