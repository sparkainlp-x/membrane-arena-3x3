# Membrane Arena 3x3 + Yin-Yang Shield

**SYNTHETIC classical software simulation.** This repository models a 3x3 lattice of 32-dimensional density-matrix cells that carry the name "holographic universe", with two independent shield components:

- **Yang Observer Shield**: read-only measurement and stability telemetry.
- **Yin Calibrator Shield**: one bounded global feedback signal, plus an optional sinusoid labelled "Schumann-mode".

Everything runs as ordinary NumPy linear algebra on a classical computer. It is **not** a physical holographic-universe simulator, a quantum computer or quantum hardware simulation of any real device, consciousness research, a medical device, or a control system. "Holographic universe", "Yin", "Yang" and "Schumann" are names and metaphors only. The 7.83 Hz signal is an arbitrary sinusoid: this project makes no physical, physiological or biological claim about it. All numbers below are synthetic.

Status: version 0.1.0, unreleased. There is no tagged release and no DOI yet; this line is updated when that changes.

## Relation to membrane-arena

[sparkainlp-x/membrane-arena](https://github.com/sparkainlp-x/membrane-arena) is a separate project: a zero-dependency browser game about three coupled Oja-rule membranes. This repository is a different Python model (a 3x3 lattice of density matrices). It shares a name and the SYNTHETIC framing, and no code. Nothing here was ported from, or checked against, the game or the crystallization-contest baselines.

## Dependencies

Runtime: NumPy only. Development: pytest. Most of the author's other repositories are standard-library only; this one is the NumPy exception because it needs eigendecompositions of 32x32 complex matrices.

## Model

Each cell is a density matrix `rho[i, j]` in C^32 x C^32, with `rho = rho^dagger`, `rho >= 0` and `trace(rho) = 1`. The nine cells are arranged as:

```text
A1: M11 M12 M13
A2: M21 M22 M23
A3: M31 M32 M33
```

One update of a cell applies, in this order: an Euler step of the commutator with a fixed random Hermitian matrix, a damping filter (`apply_dephasing`), a coupling term toward the four neighbours (same arena and same membrane in the other arenas), a pull toward the cell's initial reference state scaled by the calibrator signal, a projection back onto valid density matrices, and a nonlinear purity filter `rho -> rho^(1+s) / tr(rho^(1+s))` (`crystallization_mirror`).

The observer then reports, over all nine cells: mean purity (called `mean_coherence`), maximum pairwise divergence `1 - fidelity`, maximum residual (largest absolute entry of `rho - reference`), minimum purity, mean normalized entropy and a weighted penalty called `lyapunov_proxy`.

The update is transactional. A candidate lattice is rejected as a whole if any of the nine cells is not a valid density matrix, or if the residual, divergence or minimum-purity guard fails, or if a lattice parameter is outside its allowed range. A rejected update changes nothing. After `failure_limit` consecutive rejected updates (default 5) the circuit breaker latches and every later step does nothing.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
membrane-arena-3x3 --steps 200 --seed 42 --json results.json
```

With the sinusoid:

```bash
membrane-arena-3x3 --steps 200 --seed 42 \
  --schumann --schumann-hz 7.83 --schumann-amplitude 0.02 \
  --json results.json
```

The CLI prints a short summary; `--json` writes every observation and calibration action. Invalid settings, including a sinusoid frequency at or above the sampling limit `1 / (2 dt)`, are reported as usage errors.

## Measured behavior of the default settings

Measured on one machine (Python 3.13.5, NumPy 2.5.4) by `scripts/injection_sweep.py` (also written to `docs/injection_results.md`), 200 steps requested, default settings. The last two columns before the cost are the event-score totals at a fixed commit of 0.35 (the default) and of 0.75:

| seed | base steps / accepted | injected steps / accepted | injection passes of events | event-score total (commit 0.35) | event-score total (commit 0.75) | sampling cost | final status |
|---|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 11 / 6 | 3 of 8 | -40.2 | -184.7 | 15.2 | circuit_breaker_open |
| 0 | 9 / 4 | 10 / 5 | 1 of 6 | 99.4 | -333.1 | 13.6 | circuit_breaker_open |
| 1 | 8 / 3 | 10 / 5 | 2 of 7 | 37.1 | -269.9 | 15.2 | circuit_breaker_open |
| 2 | 8 / 3 | 11 / 6 | 3 of 8 | -40.2 | -184.7 | 15.2 | circuit_breaker_open |
| 3 | 9 / 4 | 11 / 6 | 2 of 7 | 12.0 | -242.5 | 13.6 | circuit_breaker_open |

Ablation, the same loop with the mirror left at the stock 0.10 (smaller `dt` only):

| seed | base steps / accepted | injected steps / accepted | injection passes of events | event-score total (commit 0.35) | event-score total (commit 0.75) | sampling cost | final status |
|---|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 9 / 4 | 1 of 6 | 81.8 | -316.7 | 12.8 | circuit_breaker_open |
| 0 | 9 / 4 | 9 / 4 | 0 of 5 | 151.6 | -390.9 | 12.0 | circuit_breaker_open |
| 1 | 8 / 3 | 9 / 4 | 1 of 6 | 106.9 | -344.1 | 14.4 | circuit_breaker_open |
| 2 | 8 / 3 | 9 / 4 | 1 of 6 | 81.8 | -316.7 | 12.8 | circuit_breaker_open |
| 3 | 9 / 4 | 9 / 4 | 0 of 5 | 151.6 | -390.9 | 12.0 | circuit_breaker_open |

### Multi-seed sweep (seeds 0 to 49)

`python scripts/injection_multiseed.py --write` runs 50 seeds (200 steps requested) for the default setting and the dt-only ablation, scores the commit policies under both the engine's two-forecast scoring and the per-attempt scoring, and reports means with 95% bootstrap intervals, pass rates, breaker counts and paired differences against the stock engine. The full tables and the "Honest limits" are in `docs/injection_results.md` (section between the multiseed markers); the raw per-seed runs are in `experiments/results/multiseed_runs.jsonl`. In short, as measured: the injection adds accepted steps (see the paired difference in that table), the breaker still latched in every run, and the sign of a policy's ledger score depends on the scoring rule. It does not cure the arena.

### Commit policy and regime dependence

The commit never changes acceptance, so the dynamics are identical under every policy; only the ledger differs. The table below is the stock regime, measured on this machine (SYNTHETIC). Scores are event-score totals in points (two forecasts per event, minus 0.8 per variant).

| seed | steps / accepted | passes of events | fixed 0.75 | fixed 0.35 | adaptive from 0.35 | adaptive from 0.75 | adaptive final commit (from 0.35) |
|---|---|---|---|---|---|---|---|
| 42 | 11 / 6 | 3 of 8 | -184.7 | -40.2 | -82.3 | -62.6 | 0.367 |
| 0 | 10 / 5 | 1 of 6 | -333.1 | 99.4 | 85.9 | -101.6 | 0.240 |
| 1 | 10 / 5 | 2 of 7 | -269.9 | 37.1 | -23.9 | -103.7 | 0.309 |
| 2 | 11 / 6 | 3 of 8 | -184.7 | -40.2 | -82.3 | -62.6 | 0.367 |
| 3 | 11 / 6 | 2 of 7 | -242.5 | 12.0 | -25.3 | -74.0 | 0.309 |
| total | | | -1215.0 | 68.0 | -127.9 | -404.6 | |

The pass rate over these 36 events is 11 of 36 (0.31). A fixed 0.35 sits close to that rate and scores positive on these five seeds (+68.0 in total); a fixed 0.75 is overconfident and scores -1215.0. The 0.35 default was set from these same seeds, so that score is in-sample, not a held-out result, and it does not hold up on more seeds: over seeds 0 to 49 the mean two-forecast score of a fixed 0.35 is -5.0 per run with a 95% interval that includes zero, and on the 100 disjoint confirm seeds of the pre-registered analysis it is +11.2 per run with an interval that also includes zero (see the multi-seed section below and `experiments/results/prereg_tables.md`), although its first-forecast Brier score is far better than that of a fixed 0.75. The adaptive commit scores below the fixed 0.35 in this regime because it starts without knowing the rate (a burn-in cost); its value is that it does not need the rate in advance. Started from 0.75 it scores better than fixed 0.75 (-404.6 against -1215.0), but the total is still negative on this short run.

**0.35 is a default for the stock regime only.** Under a different parameter set the pass rate differs, and a fixed default cannot be right in every regime. Using the genome supplied with the design note (coupling 0.0339, dephasing 0.2871, mirror_strength 0.1725, dt 0.0128, `first_factor` 0.2555, `mirror_variant` 0.0; seeds 42, 0, 1), measured here:

| seed | steps / accepted | passes of events | fixed 0.75 | fixed 0.35 | adaptive from 0.35 | adaptive from 0.75 | adaptive final commit (from 0.35) |
|---|---|---|---|---|---|---|---|
| 42 | 58 / 53 | 50 of 55 | 3264.3 | -3288.2 | 3459.2 | 4009.5 | 0.871 |
| 0 | 87 / 82 | 80 of 85 | 5512.2 | -5397.3 | 6388.7 | 6975.4 | 0.915 |
| 1 | 56 / 51 | 48 of 53 | 3143.3 | -3173.7 | 3300.2 | 3845.9 | 0.867 |
| total | | | 11919.9 | -11859.1 | 13148.1 | 14830.8 | |

In that regime about 0.9 of the variants pass. A fixed 0.35 is underconfident and scores negative (-11859.1 in total); a fixed 0.75 scores +11919.9. The adaptive commit from 0.35 ends near 0.87 to 0.92 and scores +13148.1, above both fixed commits, and adaptive from 0.75 scores +14830.8. The breaker still latched in all three evolved-genome runs. This is a result for these three seeds only; the genome was supplied with a design note and was not re-evolved here.

**Not reproduced here.** The design note also describes a true phase-damping experiment and fitness values from the evolution run (fitness, mean executed and accepted counts, a held-out seed result). The phase-damping code was not supplied and the fitness function was not available, so neither is part of this repository and no claim is made about them. The injection does not cure the breaker in any regime measured here.

These numbers describe this synthetic model and these settings only. Regenerate them with `python scripts/injection_sweep.py`.

## Guard rules

- The observer works on read-only views, so an accidental write inside it raises instead of changing the lattice (`tests/test_observer.py`).
- The calibrator signal is clipped to +/- 0.05 and every action is logged, with its accept or reject reason.
- A failed validity, residual, divergence, purity or parameter check rejects the complete update, and the lattice is bit-identical afterwards (`tests/test_transaction.py`).
- After the configured number of consecutive rejections the breaker latches and later steps do nothing (`tests/test_breaker.py`).

These are software checks inside a toy model. They are not safety mechanisms for any real system.

## Tests

```bash
pytest -q
```

114 tests (63 for the stock arena, 24 for the opt-in injection, 27 for the experiment helpers and the multi-seed script): validity of the density-matrix primitives, observer read-only behavior on locked arrays, all-or-nothing rejection, breaker behavior under forced failures, determinism by seed, configuration limits, the CLI, and the pinned default behavior.

## Credit

Jean-François Brisson, Spark AI NLP, Fredericton, New Brunswick, Canada.

## License

Copyright (c) 2026 Jean-François Brisson, Spark AI NLP.

Available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).
