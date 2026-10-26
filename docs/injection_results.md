# Injection sweep (SYNTHETIC, measured by scripts/injection_sweep.py)

Python 3.13.5, NumPy 2.5.4, 200 steps requested, default `InjectionConfig` (3 attempts, fixed commit 0.35). Injection RNG: default_rng(seed + 10000007), one uniform(0.8, 1.2) draw per attempt.

## Default injection (dt shrink + mirror softened to 0.0)

| seed | base steps / accepted | injected steps / accepted | injection passes of events | event-score total (commit 0.35) | event-score total (commit 0.75) | sampling cost | final status |
|---|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 11 / 6 | 3 of 8 | -40.2 | -184.7 | 15.2 | circuit_breaker_open |
| 0 | 9 / 4 | 10 / 5 | 1 of 6 | 99.4 | -333.1 | 13.6 | circuit_breaker_open |
| 1 | 8 / 3 | 10 / 5 | 2 of 7 | 37.1 | -269.9 | 15.2 | circuit_breaker_open |
| 2 | 8 / 3 | 11 / 6 | 3 of 8 | -40.2 | -184.7 | 15.2 | circuit_breaker_open |
| 3 | 9 / 4 | 11 / 6 | 2 of 7 | 12.0 | -242.5 | 13.6 | circuit_breaker_open |

## Ablation: dt shrink only (mirror left at the stock 0.10)

| seed | base steps / accepted | injected steps / accepted | injection passes of events | event-score total (commit 0.35) | event-score total (commit 0.75) | sampling cost | final status |
|---|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 9 / 4 | 1 of 6 | 81.8 | -316.7 | 12.8 | circuit_breaker_open |
| 0 | 9 / 4 | 9 / 4 | 0 of 5 | 151.6 | -390.9 | 12.0 | circuit_breaker_open |
| 1 | 8 / 3 | 9 / 4 | 1 of 6 | 106.9 | -344.1 | 14.4 | circuit_breaker_open |
| 2 | 8 / 3 | 9 / 4 | 1 of 6 | 81.8 | -316.7 | 12.8 | circuit_breaker_open |
| 3 | 9 / 4 | 9 / 4 | 0 of 5 | 151.6 | -390.9 | 12.0 | circuit_breaker_open |

## Commit policy, stock regime (score in points; dynamics identical across policies)

| seed | steps / accepted | passes of events | fixed 0.75 | fixed 0.35 | adaptive from 0.35 | adaptive from 0.75 | adaptive final commit (from 0.35) |
|---|---|---|---|---|---|---|---|
| 42 | 11 / 6 | 3 of 8 | -184.7 | -40.2 | -82.3 | -62.6 | 0.367 |
| 0 | 10 / 5 | 1 of 6 | -333.1 | 99.4 | 85.9 | -101.6 | 0.240 |
| 1 | 10 / 5 | 2 of 7 | -269.9 | 37.1 | -23.9 | -103.7 | 0.309 |
| 2 | 11 / 6 | 3 of 8 | -184.7 | -40.2 | -82.3 | -62.6 | 0.367 |
| 3 | 11 / 6 | 2 of 7 | -242.5 | 12.0 | -25.3 | -74.0 | 0.309 |
| total | | | -1215.0 | 68.0 | -127.9 | -404.6 | |

## Commit policy, evolved genome supplied with the design note (a different regime)

Genome: coupling 0.0339, dephasing 0.2871, mirror_strength 0.1725, dt 0.0128, first_factor 0.2555, mirror_variant 0.0. Seeds 42, 0, 1.

| seed | steps / accepted | passes of events | fixed 0.75 | fixed 0.35 | adaptive from 0.35 | adaptive from 0.75 | adaptive final commit (from 0.35) |
|---|---|---|---|---|---|---|---|
| 42 | 58 / 53 | 50 of 55 | 3264.3 | -3288.2 | 3459.2 | 4009.5 | 0.871 |
| 0 | 87 / 82 | 80 of 85 | 5512.2 | -5397.3 | 6388.7 | 6975.4 | 0.915 |
| 1 | 56 / 51 | 48 of 53 | 3143.3 | -3173.7 | 3300.2 | 3845.9 | 0.867 |
| total | | | 11919.9 | -11859.1 | 13148.1 | 14830.8 | |

The injection never changes a guard, threshold or the breaker. Injected runs in the first two tables ending with the breaker open: 10 of 10; the injection does not cure the arena. 'Injection score' is the sum of Crystallize points (100 * (1 - 4 * Brier)): per event 0.5 * points(first commit) + 0.5 * points(last commit) minus 0.8 per variant tried (two forecasts per event, not every variant). The commit changes only this ledger. A fixed commit scores well only where it is near the regime's pass rate; the adaptive commit tracks the rate but pays a burn-in cost.

<!-- multiseed:start -->
## Multi-seed sweep, seeds 0 to 49 (50 seeds, stock regime)

Measured by `scripts/injection_multiseed.py` on Python 3.13.5, NumPy 2.5.4, 200 steps requested per run, 3 injection attempts, injection RNG `default_rng(seed + 10000007)`. Intervals are 95% percentile bootstrap intervals over seeds (10,000 resamples). One injected run per seed and setting serves every commit policy, because the commit changes only the ledger.

Stock engine (no injection): breaker open in 50/50 runs, steps 8.5 [8.4, 8.6], accepted steps 3.5 [3.4, 3.6] (mean [95% CI]).

### Steps, acceptance and breaker, injected versus stock (paired by seed)

| setting | steps injected | accepted injected | accepted, injected minus stock | steps, injected minus stock | breaker open (Wilson 95%) | events per run | pooled pass rate [cluster 95% CI] (passes / events) |
|---|---|---|---|---|---|---|---|
| default (dt shrink + mirror 0.0) | 11.3 [10.9, 11.7] | 6.2 [5.8, 6.5] | 2.66 [2.30, 3.02] (range 0 to 7) | 2.78 [2.40, 3.18] | 50/50 (0.929 to 1.000) | 7.8 [7.4, 8.2] | 0.342 [0.309, 0.373] (133 / 389) |
| ablation: dt shrink only (mirror 0.10) | 9.2 [9.1, 9.5] | 4.2 [4.0, 4.3] | 0.68 [0.54, 0.80] (range 0 to 1) | 0.74 [0.56, 0.94] | 50/50 (0.929 to 1.000) | 5.7 [5.6, 5.9] | 0.118 [0.097, 0.138] (34 / 287) |

### Ledger score per run by commit policy, mean [95% CI]

Two-forecast scoring is the engine's (0.5 points of the first commit + 0.5 points of the last commit - 0.8 per variant tried). Per-attempt scoring (every variant scored, cost not subtracted) is the scoring the first merged version used; it is computed offline from the same events and shown for comparison. The last column is the mean first-forecast Brier score (lower is better) over runs that had events.

default (dt shrink + mirror 0.0):

| commit policy | two-forecast score | per-attempt score | mean first-forecast Brier | runs with two-forecast score < 0 |
|---|---|---|---|---|
| fixed 0.35 | -5.0 [-28.9, 18.2] | 424.7 [395.7, 455.8] | 0.220 [0.210, 0.229] | 23/50 |
| fixed 0.75 | -229.0 [-257.7, -201.3] | -1096.4 [-1153.4, -1046.2] | 0.400 [0.385, 0.416] | 48/50 |
| adaptive from 0.35 | -35.0 [-56.8, -10.9] | 14.2 [-74.6, 107.2] | 0.245 [0.231, 0.257] | 42/50 |
| adaptive from 0.75 | -72.5 [-83.2, -59.7] | -753.0 [-835.4, -668.8] | 0.313 [0.310, 0.317] | 48/50 |

ablation: dt shrink only (mirror 0.10):

| commit policy | two-forecast score | per-attempt score | mean first-forecast Brier | runs with two-forecast score < 0 |
|---|---|---|---|---|
| fixed 0.35 | 112.3 [103.7, 121.3] | 457.2 [439.9, 477.3] | 0.156 [0.149, 0.162] | 0/50 |
| fixed 0.75 | -351.6 [-364.9, -340.3] | -1058.1 [-1098.7, -1025.6] | 0.506 [0.496, 0.517] | 50/50 |
| adaptive from 0.35 | 130.3 [110.2, 151.7] | 517.5 [471.4, 568.0] | 0.144 [0.128, 0.159] | 0/50 |
| adaptive from 0.75 | -85.4 [-89.7, -81.2] | -319.3 [-360.2, -278.5] | 0.312 [0.309, 0.316] | 50/50 |

### Honest limits

- This is one toy model (a synthetic 3x3 lattice of 32-dimensional density matrices) at its published default parameters, on one machine and one NumPy. Nothing here says anything about a physical system.
- The 50 seeds are a convenience set (0 to 49) that includes the five seeds on which the 0.35 default was chosen. The 0.35 rows are therefore partly in-sample. The pre-registered analysis uses disjoint seeds.
- The sweep shows what trying shrunk variants after a rejection changes in this model: the accepted-step and steps columns above, and a ledger that measures calibration. It does not show that the injection cures the arena. Breaker open at the end of the run, injected: default (dt shrink + mirror 0.0) 50/50, ablation: dt shrink only (mirror 0.10) 50/50; stock 50/50.
- Forecasts do not use event features, so the ledger measures calibration to a base rate, not event-level skill. A score depends on how close the commit is to the pass rate and on the scoring rule chosen. For the default setting and a fixed 0.35 commit the mean score is 424.7 under per-attempt scoring and -5.0 [-28.9, 18.2] under the engine's two-forecast scoring (per-attempt scoring counts every failed variant as a separate miss and subtracts no cost; the two-forecast rule scores two forecasts against the event outcome and subtracts 0.8 per variant), so a score is not an outcome measure by itself and the sign of a policy's score depends on the scoring rule.
- Within-run events are not independent (the last events of every run are failures by construction, because the breaker needs consecutive rejections), so the pooled pass rate is a descriptive rate, and the intervals above resample seeds, not events.
- The intervals are uncorrected for multiplicity and are for magnitude, not for a decision.
<!-- multiseed:end -->
