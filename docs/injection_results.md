# Injection sweep (SYNTHETIC, measured by scripts/injection_sweep.py)

Python 3.12.14, NumPy 2.5.4, 200 steps requested, default `InjectionConfig` (3 attempts, fixed commit 0.35).

## Default injection (dt shrink + mirror softened to 0.0)

| seed | base steps / accepted | injected steps / accepted | injection passes of events | injection score (points) | sampling cost | final status |
|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 11 / 6 | 3 of 8 | 346.7 | 15.2 | circuit_breaker_open |
| 0 | 9 / 4 | 10 / 5 | 1 of 6 | 484.7 | 13.6 | circuit_breaker_open |
| 1 | 8 / 3 | 9 / 4 | 1 of 6 | 396.9 | 12.8 | circuit_breaker_open |
| 2 | 8 / 3 | 12 / 7 | 4 of 9 | 409.9 | 17.6 | circuit_breaker_open |
| 3 | 9 / 4 | 11 / 6 | 2 of 7 | 327.9 | 13.6 | circuit_breaker_open |

## Ablation: dt shrink only (mirror left at the stock 0.10)

| seed | base steps / accepted | injected steps / accepted | injection passes of events | injection score (points) | sampling cost | final status |
|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 9 / 4 | 1 of 6 | 396.9 | 12.8 | circuit_breaker_open |
| 0 | 9 / 4 | 9 / 4 | 0 of 5 | 465.9 | 12.0 | circuit_breaker_open |
| 1 | 8 / 3 | 9 / 4 | 1 of 6 | 529.1 | 14.4 | circuit_breaker_open |
| 2 | 8 / 3 | 9 / 4 | 1 of 6 | 396.9 | 12.8 | circuit_breaker_open |
| 3 | 9 / 4 | 9 / 4 | 0 of 5 | 465.9 | 12.0 | circuit_breaker_open |

## Commit policy, stock regime (score in points; dynamics identical across policies)

| seed | steps / accepted | passes of events | fixed 0.75 | fixed 0.35 | adaptive from 0.35 | adaptive from 0.75 | adaptive final commit (from 0.35) |
|---|---|---|---|---|---|---|---|
| 42 | 11 / 6 | 3 of 8 | -970.3 | 346.7 | -115.1 | -727.4 | 0.367 |
| 0 | 10 / 5 | 1 of 6 | -1120.3 | 484.7 | 448.1 | -485.2 | 0.240 |
| 1 | 9 / 4 | 1 of 6 | -964.1 | 396.9 | 360.3 | -328.9 | 0.240 |
| 2 | 12 / 7 | 4 of 9 | -1128.1 | 409.9 | -368.3 | -1108.2 | 0.415 |
| 3 | 11 / 6 | 2 of 7 | -889.1 | 327.9 | 109.6 | -443.0 | 0.309 |
| total | | | -5071.9 | 1966.1 | 434.5 | -3092.7 | |

## Commit policy, evolved genome supplied with the design note (a different regime)

Genome: coupling 0.0339, dephasing 0.2871, mirror_strength 0.1725, dt 0.0128, first_factor 0.2555, mirror_variant 0.0. Seeds 42, 0, 1.

| seed | steps / accepted | passes of events | fixed 0.75 | fixed 0.35 | adaptive from 0.35 | adaptive from 0.75 | adaptive final commit (from 0.35) |
|---|---|---|---|---|---|---|---|
| 42 | 59 / 54 | 51 of 56 | 2553.1 | -2920.9 | 2016.3 | 2435.5 | 0.873 |
| 0 | 86 / 81 | 79 of 84 | 4420.3 | -4720.7 | 4183.3 | 4658.3 | 0.914 |
| 1 | 56 / 51 | 48 of 53 | 2171.9 | -2626.1 | 1458.2 | 1842.5 | 0.867 |
| total | | | 9145.3 | -10267.7 | 7657.9 | 8936.3 | |

The injection never changes a guard, threshold or the breaker. Injected runs in the first two tables ending with the breaker open: 10 of 10; the injection does not cure the arena. 'Injection score' is the sum of Crystallize points (100 * (1 - 4 * Brier)) over every variant tried. The commit changes only this ledger. A fixed commit scores well only where it is near the regime's pass rate; the adaptive commit tracks the rate but pays a burn-in cost.
