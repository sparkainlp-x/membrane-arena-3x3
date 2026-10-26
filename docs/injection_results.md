# Injection sweep (SYNTHETIC, measured by scripts/injection_sweep.py)

Python 3.13.5, NumPy 2.5.4, 200 steps requested, default `InjectionConfig` (3 attempts, commit p = 0.75).

## Default injection (dt shrink + mirror softened to 0.0)

| seed | base steps / accepted | injected steps / accepted | injection passes of events | injection score (points) | sampling cost | final status |
|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 11 / 6 | 3 of 8 | -970.3 | 15.2 | circuit_breaker_open |
| 0 | 9 / 4 | 10 / 5 | 1 of 6 | -1120.3 | 13.6 | circuit_breaker_open |
| 1 | 8 / 3 | 9 / 4 | 1 of 6 | -964.1 | 12.8 | circuit_breaker_open |
| 2 | 8 / 3 | 12 / 7 | 4 of 9 | -1128.1 | 17.6 | circuit_breaker_open |
| 3 | 9 / 4 | 11 / 6 | 2 of 7 | -889.1 | 13.6 | circuit_breaker_open |

## Ablation: dt shrink only (mirror left at the stock 0.10)

| seed | base steps / accepted | injected steps / accepted | injection passes of events | injection score (points) | sampling cost | final status |
|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 9 / 4 | 1 of 6 | -964.1 | 12.8 | circuit_breaker_open |
| 0 | 9 / 4 | 9 / 4 | 0 of 5 | -1039.1 | 12.0 | circuit_breaker_open |
| 1 | 8 / 3 | 9 / 4 | 1 of 6 | -1196.9 | 14.4 | circuit_breaker_open |
| 2 | 8 / 3 | 9 / 4 | 1 of 6 | -964.1 | 12.8 | circuit_breaker_open |
| 3 | 9 / 4 | 9 / 4 | 0 of 5 | -1039.1 | 12.0 | circuit_breaker_open |

The injection never changes a guard, threshold or the breaker. Injected runs ending with the breaker open: 10 of 10; the injection does not cure the arena. 'Injection score' is the sum of Crystallize points (100 * (1 - 4 * Brier)) over every variant tried; it is negative because most variants fail the guards while the commit is 0.75.
