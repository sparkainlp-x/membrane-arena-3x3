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
