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

Measured on one machine (Python 3.13.5, NumPy 2.5.4), seed 42, 200 steps requested, about 0.3 s wall time per run:

| Run | Steps executed | Accepted | Rejected | Status | Final max residual | Final min purity | Final max divergence |
|---|---|---|---|---|---|---|---|
| default | 8 | 3 | 5 | circuit_breaker_open | 0.1507 | 0.1191 | 0.4804 |
| `--schumann` | 8 | 3 | 5 | circuit_breaker_open | 0.1508 | 0.1191 | 0.4804 |

Across seeds 0 to 19 the default run latched the breaker in all 20 runs, after 8 or 9 steps, and every rejection was `residual_above_target`. Changing only the lattice parameters on the command line (dephasing 0, mirror 0, smaller `dt`, and combinations tried on 5 seeds) still latched the breaker after 6 to 12 steps. These numbers describe this synthetic model and these defaults only; they say nothing about any physical system. The test suite pins the default behavior (see `tests/test_default_behavior.py`).

## Scientific upgrade (configurable guards, adaptive retry, benchmark)

The stock defaults above are kept for reproducibility, and the upgrade below is layered on top of them without changing a guard value by default:

- **Configurable guards.** `GuardConfig` holds `residual_target` (0.20), `divergence_min` (0.02), `divergence_max` (0.60), `purity_floor` (0.05) and `entropy_target` (0.5); `EngineConfig.failure_limit` (5) is logged with them. Every run summary's `guards` object logs all six, and the CLI exposes them as `--residual-target`, `--divergence-min`, `--divergence-max`, `--purity-floor`, `--entropy-target` and `--failure-limit`.
- **Bounded adaptive retry (default on).** On a rejection the stock engine now tries up to `max_retries` (3) variants — `dt` × 0.5, 0.25, 0.125, mirror strength × 0.5, 0, 0, coupling × 0.5, 0.25, 0 — and counts **one** failure only after all variants fail. A run can therefore no longer stop from repeated identical rejections: each counted failure carries distinct repair attempts. Use `--no-adaptive-retry` to reproduce the stock death spiral. Measured, seed 42, 200 steps requested: with retry 12 steps / 7 accepted / 4 repair accepts; without retry 8 steps / 3 accepted.
- **Rejection telemetry.** Every counted failure is recorded in `summary["rejections"]` with the failed guard, the offending cells (per guard), the metric values at the candidate, the thresholds in effect, and the repair attempts tried.
- **Metric names.** `mean_coherence` is now `mean_purity` and `lyapunov_proxy` is now `stability_penalty` (the old names remain as read-only properties on the observation object, but `to_dict()` and JSON output use the new names). True off-diagonal measures are added: `mean_off_diagonal_coherence` and `max_off_diagonal_coherence` (mean / max |rho[i, j]|, i != j).
- **`divergence_min` is now a real check.** A lattice whose mean pairwise divergence falls below it is rejected as `divergence_below_minimum` (collapsed/synchronised). The default 0.02 is far below any divergence a random lattice shows, so stock behaviour is unchanged.
- **Improved operators, opt-in.** `LatticeConfig.operator_mode` is `"legacy"` (default, published update exactly) or `"exact"`: exact unitary conjugation `exp(-i H dt)` (preserves trace and purity to numerical precision) and trace-preserving off-diagonal phase damping `exp(-gamma |i - j|)` (diagonal untouched, no renormalisation). The stages are independently testable methods on the lattice (`unitary_stage`, `dissipative_stage`, `interaction_delta`, `mirror_stage`). Measured, seed 42: exact mode first fails on `purity_too_low`, not on `residual_above_target` — the operators change the failure mode, they do not by themselves cure the breaker.
- **API notes for subclasses and mocks.** `propose_step` now accepts `dt`, `mirror_strength`, `coupling`, `dephasing` and `operator_mode` overrides; any lattice subclass that overrides it (for example the phase-damping experiment) must accept those keywords, even if it ignores them. Guard thresholds live in `GuardConfig`; the observer's plain threshold attributes are read-only mirrors of it, so setting them after construction no longer changes a check. The stock adaptive retry subsumes the Crystallize injection on seed 42 (stock 12 steps / 7 accepted vs injected 11 / 6 with retry on; with retry off the injection still extends 8 -> 11) — the injection's ledger value is unchanged, its step-count advantage is not.
- **Benchmark.** `scripts/benchmark.py` runs seed sweeps (default 0–99), dt / coupling / operator-mode sweeps, survival curves and metric distributions, and writes deterministic CSV/JSON (no timestamps, so a re-run is byte-identical) plus optional survival-curve plots. Example: `python scripts/benchmark.py --out-dir benchmark_results --seeds 0-99 --no-plots`.

The upgrade does not change what the model is: synthetic classical software, no quantum hardware, no physical holography, no biological, medical or consciousness claim. The Crystallize injection subclass is unchanged in behaviour — it overrides `step()` and still runs its own variant loop — and its tests still pass against the upgraded engine.

## Known limits

These are properties of the model as published, not bugs that were fixed. Statements marked "exploratory" come from throwaway experiments that are not part of the test suite.

1. **The default run stops after about 8 steps without retry, 12 with it.** The default dynamics sharpen the cells (minimum purity goes from about 0.07 to 0.12 in 8 steps) faster than the guards allow. Without adaptive retry, the first rejection leaves the state unchanged, the calibrator signal depends only on the state, so the retry is the same proposal and is rejected again for the same reason; five identical rejections latch the breaker, and with the sinusoid off, "5 consecutive failures" is effectively "1 failure". With the default adaptive retry (see Scientific upgrade above), each counted failure follows three distinct repair attempts, and seed 42 reaches 12 steps / 7 accepted before the breaker latches. The breaker still latches: retry buys steps, it does not cure the drift.
2. **The guards sit close to the random starting states.** A random 32-dimensional state has purity near 2/32, about 0.06, against a purity floor of 0.05, and pairwise divergence near 0.4 to 0.5 against a maximum of 0.6. The residual guard is the largest entry of a trace-one matrix, so 0.2 is large for mixed states and is crossed as soon as cells sharpen. Exploratory: with the residual guard widened to 1.0 and the divergence maximum to 0.99, runs on seeds 0 to 4 lasted 20 to 30 steps, and the seed-0 run ended with minimum purity 0.80 and maximum divergence 0.99, that is, the cells collapse toward different nearly pure states, which is the opposite of the stated target of bounded divergence.
3. **`apply_dephasing` is not phase damping.** It multiplies entries by `exp(-gamma (i + j))` and renormalizes, which also damps the diagonal and shifts weight to low basis indices. It acts as a purifier. The legacy mode keeps it as published, with a docstring that says what it does. The exact mode replaces it with `apply_phase_damping_tp`, a trace-preserving damping of off-diagonal entries only, `exp(-gamma |i - j|)`; exploratory: that change alone changed the default run to 18 steps (breaker latched on the divergence guard).
4. **The commutator term is an explicit Euler step, not a unitary, in legacy mode.** The Hamiltonians have spectral norm near 10.5 and `dt` is 0.05, so `dt * ||H||` is about 0.5. The projection step hides the resulting loss of norm preservation. Exploratory: replacing it with the exact `exp(-i H dt)` conjugation did not change the 8-step outcome, so the early stop is not an integrator artifact. The exact mode now ships that replacement as `exact_unitary_step`; combined with trace-preserving damping it latched after 7 steps on the purity floor in the exploratory run, and after 8 steps (first failure `purity_too_low`) in the upgraded engine on seed 42.
5. **The calibrator is saturated and weak.** Its coherence target (0.85) is far above the purity of these states (about 0.06 to 0.15), so its feedback sits at the clip (+0.05) at every step of the default runs (`test_default_feedback_is_saturated_at_the_clip`). Its only effect is a pull toward the initial state of strength `signal * dt`, about 0.0025 per step, against a purity filter (exponent 1.1) applied at every step. It never proposes new coupling, dephasing or mirror values: the `coupling`, `dephasing` and `mirror_strength` fields of each logged action record the values in effect, which do not change. The parameter-bound check therefore only catches out-of-range settings given by the user (for example `--coupling 0.5`), which then reject every step.
6. **The 7.83 Hz sinusoid is nearly inert.** It has amplitude 0.02 and is added to a feedback that is already at the clip of 0.05, so the total is clipped and only the negative half of each cycle survives. At `dt = 0.05` the signal is sampled at 20 Hz, about 2.55 samples per cycle; that is below the Nyquist limit of 10 Hz, so it does not alias, but it is a crude sampling of a sinusoid, and at those sample points it equals the negated 12.17 Hz sinusoid. Configurations at or above the limit are rejected. The default runs with and without the sinusoid agree to better than 0.001 in the final metrics (`test_the_sinusoid_barely_changes_a_default_run`). The time unit of `dt` is a simulation unit; "Hz" has no physical meaning here.
7. **The coupling term is almost inert.** Below the resonance threshold (0.9) the drive is negative, so neighbours repel; random states have fidelity near 0.55, and the largest per-step entry change from the coupling term is about 5e-6 against about 2.6e-2 for the damping filter. Final metrics change in the third or fourth decimal between coupling 0 and the maximum 0.10.
8. **The old name `mean_coherence` was mean purity**, not an off-diagonal coherence measure, and duplicates `min_purity` in what it measures. It is now renamed `mean_purity`, and true off-diagonal measures (`mean_off_diagonal_coherence`, `max_off_diagonal_coherence`) are reported alongside it.
9. **The old name `lyapunov_proxy` was not a Lyapunov function.** It is a weighted penalty with an entropy target (now configurable as `entropy_target`, default 0.5). It is now renamed `stability_penalty`. Exploratory: it rose in 9 of 24 consecutive steps in one relaxed run, so nothing guarantees it decreases.
10. **`divergence_min` was unused** by any check. It is now a real check (`divergence_below_minimum` on mean pairwise divergence); the default 0.02 does not change stock behaviour.
11. **One seed means one lattice.** Results from a single seed describe that random instance. Only seeds 0 to 19 were tried for the default settings.
12. **Platforms.** The 63 tests passed locally on Python 3.10, 3.11, 3.12 and 3.13 on one Linux machine, and CI runs 3.10 to 3.13 on Ubuntu. Other operating systems, other NumPy versions and run time on other hardware were not tested.

## Crystallize injection (opt-in)

**SYNTHETIC classical software; no physical claim.** Known limit 1 says the stock engine retries the identical proposal after a rejection. `MembraneArena3x3WithInjection` (in `membrane_arena_3x3/crystallize_injection.py`) is an opt-in subclass that, on a rejected step only, tries up to `max_attempts` (default 3) variant proposals before counting the failure:

1. **Commit** a probability `p` (`first_commit`, default 0.35) that a shrunk variant passes the guards. With `adaptive=True` (default off) the first commit of each event is instead the smoothed pass rate `(passes + 4 * first_commit) / (events + 4)` over earlier events, starting at `first_commit`. The commit changes only the ledger, never whether a step is accepted.
2. **Sample** a variant: `dt` times `first_factor` (default 0.5, jittered 0.8 to 1.2 from a dedicated seeded random stream, so the main generator is never touched) and the mirror strength softened to `mirror_variant` (default 0.0). The variant factor is capped below 1.
3. **Validate** it with the engine's own, unmodified `_validate`. Only a variant that passes is committed (and logged as an accepted step); a failed variant leaves the lattice unchanged.
4. **Revise** after a miss: `p` moves halfway toward 0.5 and the step shrinks again (by 0.5 per attempt).
5. **Score** with the Crystallize formulas: Brier = (p - truth)^2, points = 100 (1 - 4 Brier), where truth is whether the variant passed. Each variant costs 0.8 in the ledger.

```python
from membrane_arena_3x3.engine import EngineConfig
from membrane_arena_3x3.crystallize_injection import run_simulation_with_injection

summary = run_simulation_with_injection(EngineConfig(steps=200, seed=42))
summary["injection_passes"], summary["injection_score_total"], summary["injections"]
```

With `InjectionConfig(max_attempts=0)` the subclass delegates to the stock `step()` and behaves exactly like the stock engine. The stock `MembraneArena3x3`, the CLI and the package exports are unchanged.

**What it does NOT do:** it does **not** cure the arena. It never changes a guard, a threshold, a parameter bound or the circuit breaker, and every injected run below still ends with the breaker open; it adds a few steps and a calibration ledger. The ledger is meant to expose miscalibration: a fixed commit of 0.75 against a usually-false outcome is overconfident and scores negative, and the Brier score says so.

Measured on one machine (Python 3.12, NumPy 2.5.4) by `scripts/injection_sweep.py` (also written to `docs/injection_results.md`), 200 steps requested, default settings (fixed commit 0.35):

| seed | base steps / accepted | injected steps / accepted | injection passes of events | injection score (points) | sampling cost | final status |
|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 11 / 6 | 3 of 8 | 346.7 | 15.2 | circuit_breaker_open |
| 0 | 9 / 4 | 10 / 5 | 1 of 6 | 484.7 | 13.6 | circuit_breaker_open |
| 1 | 8 / 3 | 9 / 4 | 1 of 6 | 396.9 | 12.8 | circuit_breaker_open |
| 2 | 8 / 3 | 12 / 7 | 4 of 9 | 409.9 | 17.6 | circuit_breaker_open |
| 3 | 9 / 4 | 11 / 6 | 2 of 7 | 327.9 | 13.6 | circuit_breaker_open |

Ablation, the same loop with the mirror left at the stock 0.10 (smaller `dt` only):

| seed | base steps / accepted | injected steps / accepted | injection passes of events | injection score (points) | sampling cost | final status |
|---|---|---|---|---|---|---|
| 42 | 8 / 3 | 9 / 4 | 1 of 6 | 396.9 | 12.8 | circuit_breaker_open |
| 0 | 9 / 4 | 9 / 4 | 0 of 5 | 465.9 | 12.0 | circuit_breaker_open |
| 1 | 8 / 3 | 9 / 4 | 1 of 6 | 529.1 | 14.4 | circuit_breaker_open |
| 2 | 8 / 3 | 9 / 4 | 1 of 6 | 396.9 | 12.8 | circuit_breaker_open |
| 3 | 9 / 4 | 9 / 4 | 0 of 5 | 465.9 | 12.0 | circuit_breaker_open |

### Commit policy and regime dependence

The commit never changes acceptance, so the dynamics are identical under every policy; only the ledger differs. The table below is the stock regime, measured on this machine (SYNTHETIC). Scores are in points, summed over all variants tried.

| seed | steps / accepted | passes of events | fixed 0.75 | fixed 0.35 | adaptive from 0.35 | adaptive from 0.75 | adaptive final commit (from 0.35) |
|---|---|---|---|---|---|---|---|
| 42 | 11 / 6 | 3 of 8 | -970.3 | 346.7 | -115.1 | -727.4 | 0.367 |
| 0 | 10 / 5 | 1 of 6 | -1120.3 | 484.7 | 448.1 | -485.2 | 0.240 |
| 1 | 9 / 4 | 1 of 6 | -964.1 | 396.9 | 360.3 | -328.9 | 0.240 |
| 2 | 12 / 7 | 4 of 9 | -1128.1 | 409.9 | -368.3 | -1108.2 | 0.415 |
| 3 | 11 / 6 | 2 of 7 | -889.1 | 327.9 | 109.6 | -443.0 | 0.309 |
| total | | | -5071.9 | 1966.1 | 434.5 | -3092.7 | |

The pass rate over these 36 events is 11 of 36 (0.31). A fixed 0.35 sits close to that rate and scores positive; a fixed 0.75 is overconfident and scores negative. The 0.35 default was set from these same seeds, so its score here is in-sample, not a held-out result. The adaptive commit scores below the fixed 0.35 in this regime because it starts without knowing the rate (a burn-in cost); its value is that it does not need the rate in advance. Started from 0.75 it ends well below 0.75 and scores better than fixed 0.75, but the total is still negative on this short run.

**0.35 is a default for the stock regime only.** Under a different parameter set the pass rate differs, and a fixed default cannot be right in every regime. Using the genome supplied with the design note (coupling 0.0339, dephasing 0.2871, mirror_strength 0.1725, dt 0.0128, `first_factor` 0.2555, `mirror_variant` 0.0; seeds 42, 0, 1), measured here:

| seed | steps / accepted | passes of events | fixed 0.75 | fixed 0.35 | adaptive from 0.35 | adaptive from 0.75 | adaptive final commit (from 0.35) |
|---|---|---|---|---|---|---|---|
| 42 | 59 / 54 | 51 of 56 | 2553.1 | -2920.9 | 2016.3 | 2435.5 | 0.873 |
| 0 | 86 / 81 | 79 of 84 | 4420.3 | -4720.7 | 4183.3 | 4658.3 | 0.914 |
| 1 | 56 / 51 | 48 of 53 | 2171.9 | -2626.1 | 1458.2 | 1842.5 | 0.867 |
| total | | | 9145.3 | -10267.7 | 7657.9 | 8936.3 | |

In that regime about 0.9 of the variants pass. A fixed 0.35 is underconfident and scores negative; the adaptive commit from 0.35 ends near 0.87 to 0.91 and scores positive. A fixed 0.75 scores higher than the adaptive commit from 0.35 here, because 0.75 happens to be closer to the true rate than 0.35 is, so the adaptive commit is not the best choice in every case; it is the choice that needs no tuning. The breaker still latched in all three evolved-genome runs. This is a result for these three seeds only; the genome was supplied with a design note and was not re-evolved here.

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

136 tests in total (86 before the scientific upgrade; the rest are the new scientific-upgrade, benchmark and hypothesis tests, plus the updated summary/transaction/breaker/injection tests): validity of the density-matrix primitives, observer read-only behavior on locked arrays, all-or-nothing rejection, breaker behavior under forced failures, determinism by seed, configuration limits, the CLI, the pinned default behavior, configurable guards, adaptive retry, trace-preserving operators, benchmark determinism, and the measured dt/coupling hypotheses. Full suite: 136 passed, 0 failed (2026-10-29, ~12 minutes, the 200-step hypothesis runs dominate).

## Credit

Jean-François Brisson, Spark AI NLP, Fredericton, New Brunswick, Canada.

## License

Copyright (c) 2026 Jean-François Brisson, Spark AI NLP.

Available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).
