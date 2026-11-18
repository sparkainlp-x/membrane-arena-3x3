# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
<!-- Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP) -->
<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Lyapunov exponent — pre-registration and report

Version 0.3.0. Local work only; no push, no tag, no release, no Zenodo, no PR.

## Mechanism (as built)

New module `membrane_arena_3x3/lyapunov.py` (stdlib + NumPy only, no new
dependency). It measures the finite-time Lyapunov exponent of the lattice
MAP — `MembraneLattice.propose_step(0.0)` + `commit`, legacy mode, stock
config (coupling 0.02, dephasing 0.5, mirror 0.10, dt 0.05) unless stated
— not of the guarded engine (guards, adaptive retry and the calibrator
would inject discontinuous accept/reject branching into the twin
distance) and not of `shields.Observation.lyapunov_proxy`, which is a
kept alias for `stability_penalty` and whose docstring says it is not a
Lyapunov function. Key functions:

- `trace_distance(rho, sigma)` — 0.5·‖ρ−σ‖₁, in [0, 1]; saturation
  `D_SAT = 1.0`.
- `twin_trace_distance(state_a, state_b)` — PRIMARY series: maximum
  over the nine cells of the trace distance between the twins.
  `mean_twin_trace_distance` — secondary series: mean over the cells.
  `twin_infidelity` — CHECK series: maximum over the cells of
  1 − fidelity.
- `perturb_toward(rho, sigma, delta0)` — trace-preserving perturbation
  `ρ' = (1−ε)ρ + εσ` with `ε = delta0 / d(ρ, σ)`, so the trace distance
  is exactly `delta0`. No projection/clipping is involved.
- `twin_run(config, seed, steps, delta0=1e-6, ...)` — builds two
  identical lattices from `seed`, applies a deterministic `setup` to
  both, perturbs **all nine cells** of the second twin by `delta0`
  (independent random directions, `perturb_seed=20260717`), steps both
  with the bare map, records the distance series and the mean pairwise
  fidelity of twin A. All-cell perturbation is deliberate: a
  single-cell pilot showed the perturbed cell's distance decays at its
  own single-cell rate whatever the lattice regime is (neighbours enter
  its update only through the O(coupling·dt) interaction), so it cannot
  see the interaction term at all.
- `fit_lambda(times, distances, ..., window_steps=25)` — least-squares
  slope of log d(t) against t = step·dt over steps 1..25 (1.25 time
  units at dt = 0.05), excluding t = 0. Horizon
  `t_λ = (1/λ)·ln(D_SAT/δ₀)` when λ > 0 (formal extrapolation — see
  limits). The 1 − fidelity check is fitted only when its series rises
  above 1e-9; for a δ₀ = 1e-6 perturbation it is second order in the
  perturbation and sits at ~1e-11, the numerical floor of the fidelity
  computation, so contracting runs report it as not fitted rather than
  as a spurious number.
- `blend_to_mean_fidelity(lattice, target)` — bisection blend of every
  cell toward cell (0, 0) to a target mean pairwise fidelity.
- `setup_matched_hamiltonians(lattice)` — all cells share cell (0, 0)'s
  Hamiltonian. `setup_trio(lattice, cells, start)` — the round-6
  re-pinned trio construction (0.85 start, matched H, matched
  references); see `scratch-round6/exp_seed1.py`.

## Frozen protocol (set from the pilot, before the final run)

- Perturbation: all 9 cells, δ₀ = 1e-6, `perturb_seed = 20260717`.
- Primary: λ from the max-trace-distance series, steps 1..25.
  Secondary: λ from the mean-trace-distance series (same window).
  Check: 1 − fidelity series (fitted only above the 1e-9 floor).
- Claim (i) setup: blend to the target mean pairwise fidelity, then
  match Hamiltonians. The matched-clock control is required: with
  distinct Hamiltonians the unitary stage rotates the cells apart in
  one step, so a lattice blended above 0.90 collapses below 0.90 after
  a single step and the above-threshold regime is unmeasurable (pilot
  1). With matched clocks the blended fidelity persists for ~20 steps
  below threshold (crossing 0.90 at step 8–25 depending on target) and
  indefinitely above it (pilot 2). Targets {0.80, 0.85, 0.88, 0.92,
  0.95, 0.98} × seeds {42, 1, 7}. Regime label: initial mean pairwise
  fidelity F₀.
- Claim (ii) setup: the round-6 re-pinned comparison set — legacy,
  row trio, 0.85 start, uniform multiplier m = 2.5 on the trio's cells
  1–2; seed 1 (excursion) vs seeds 42 and 7 (controls), plus the seed-1
  column control. Same 25-step window.
- Claim (iii) setup: matched Hamiltonians, F₀ = 0.85, seed 42 (primary)
  plus seeds 1 and 7 (replication); dt = 0.05 (25 steps) vs dt = 0.025
  (50 steps) — the same 1.25-time-unit window at both resolutions.
  Sensitivity check: same-step-count comparison (25 steps at both dt).

## Claims (pre-registered; thresholds frozen before the final run)

- (i) λ SIGN FLIP AT 0.90. PASS iff λ > 0 at every below-threshold
  point (F₀ ∈ {0.80, 0.85, 0.88}) AND λ ≤ 0 at every above-threshold
  point (F₀ ∈ {0.92, 0.95, 0.98}), on all three seeds — 18 points, all
  must satisfy the sign. Two-sided. Secondary (directional, reported
  regardless): λ decreases monotonically in F₀ across the sweep.
- (ii) SEED-1 EXCURSION. PASS iff λ(seed 1, row, m = 2.5) exceeds the
  mean of λ(seed 42, row, m = 2.5) and λ(seed 7, row, m = 2.5) by a
  margin of **≥ 0.10** per time unit.
- (iii) HALF-DT DECISION. PASS (real dynamics) iff λ(dt) and λ(dt/2)
  have the same sign AND |λ(dt/2) − λ(dt)| ≤ **0.15** per time unit on
  the primary (max) series, on seed 42, with seeds 1 and 7 agreeing in
  direction. FAIL (they disagree) decides the half-dt question the
  other way: the dt-dependence is a projection/discretisation artifact.
  Whichever way it falls is reported honestly.

## What the pilot showed (protocol-setting only, seed 42)

The pilot set the protocol above; it is not the final measurement.
Seed 42, matched H, 25-step window: λ_max = +0.50 / +0.50 / +0.46 /
+0.32 / +0.16 / −0.06 at F₀ = 0.80 / 0.85 / 0.88 / 0.92 / 0.95 / 0.98
(R² 0.82–0.94 except 0.95/0.98), monotone decreasing in F₀ with the
zero between 0.95 and 0.98 — so claim (i) as literally stated is
expected to FAIL at F₀ = 0.92 and 0.95, while the monotone directional
signature holds. The mean series never flips sign (−0.12 … −0.38). The
stock unblended seed-42 lattice (mean F₀ ≈ 0.56) has λ_max = +0.86
(R² 0.94) — pinned as a unit test. Claim (ii) pilot: λ_max = +0.23
(seed 1 row) vs +0.59 / +0.60 (seeds 42 / 7 row), gap −0.36 — expected
FAIL; the seed-1 column control is higher still (+0.92). The excursion
is an engine-level repair-ratchet phenomenon, not a more-sensitive map.
Claim (iii) pilot: λ_max = +0.50 at dt = 0.05 vs +1.36 at dt = 0.025
(gap 0.86, far above the 0.15 tolerance), mean series flips sign
(−0.16 → +0.25) — expected DISAGREE, i.e. the artifact side. This is
consistent with the round-6 stage decomposition (Euler divergence
shrinks ~1.4× per dt halving, dephasing contraction ~4×, so halving dt
tips the balance toward divergence). These pilot expectations are
recorded here so the final run cannot move the thresholds; the final
numbers are what count, including seeds 1 and 7, which the pilot did
not cover for (i) and (iii).

## Results (final measurement run, `scripts/lyapunov_sweep.py`, 2m32s)

Full series in `lyapunov_results/lyapunov_final.json`. All λ in per-time
units, 25-step window (1.25 time units), δ₀ = 1e-6. The 1 − fidelity
check never rose above its numerical floor in any run
(`lambda_check_fitted: false` everywhere), so no check fit exists to
report — as predicted in the pre-registration, it cannot corroborate
anything at δ₀ = 1e-6.

### (i) Sign flip at 0.90 — FAIL (as the pilot expected)

λ_max (R² in brackets) by seed and F₀:

| seed | 0.80 | 0.85 | 0.88 | 0.92 | 0.95 | 0.98 |
|------|------|------|------|------|------|------|
| 42 | +0.497 (0.82) | +0.499 (0.89) | +0.457 (0.94) | +0.324 (0.90) | +0.157 (0.43) | −0.059 (0.06) |
| 1 | +0.337 (0.83) | +0.247 (0.80) | +0.170 (0.63) | +0.055 (0.10) | −0.025 (0.02) | −0.117 (0.35) |
| 7 | −0.311 (0.75) | −0.446 (0.87) | −0.490 (0.92) | −0.500 (0.93) | −0.516 (0.94) | −0.534 (0.94) |

The sign flip at 0.90 does not hold. Seed 42 stays positive at 0.92
and 0.95 (zero between 0.95 and 0.98); seed 1 stays positive at 0.92
(zero between 0.92 and 0.95); seed 7 is negative at every point,
including 0.80–0.88 below threshold. The claim fails on both sides,
in opposite directions on different seeds.

What DOES hold, on all three seeds: λ_max decreases monotonically in
F₀ (42: +0.50 → −0.06; 1: +0.34 → −0.12; 7: −0.31 → −0.53), and
λ_mean does the same (42: −0.12 → −0.38; 1: −0.13 → −0.47; 7: −0.93
→ −0.97). The interaction term's sign flip at 0.90 shows up as a
level shift — less contraction (or outright growth) below threshold,
more contraction above it — not as a sign flip of λ itself, because
the interaction (coupling 0.02) competes with seed-dependent
contraction from the dephasing/mirror/projection stages, and the
balance point is seed-dependent. The crossing-truncated sensitivity
fits agree on the signs (e.g. seed 42: +0.50 / +0.34 / +0.33 at
F₀ = 0.80 / 0.85 / 0.88; seed 7: −0.07 / −0.21 / −0.29). Horizons
where λ > 0 are formal only: 28–88 time units on seed 42, 41–249 on
seed 1 (the 249 at F₀ = 0.92 rests on R² = 0.10 and is meaningless as
a prediction); the regime itself lasts ~1 time unit and d only reaches
~2–3e-6 in the window, so saturation is never approached.

### (ii) Seed-1 excursion — FAIL

λ_max, 25 steps: seed 1 row m = 2.5 **+0.232** (R² 0.21 — the distance
plateaus after step 11, so the log-linear fit is poor) vs controls
seed 42 row **+0.594** (R² 0.39) and seed 7 row **+0.596** (R² 0.84).
Control mean +0.595; seed 1 is BELOW it by 0.363, against the
pre-registered requirement to EXCEED it by ≥ 0.10. The seed-1 column
control is higher still (+0.919, R² 0.99), and the m = 1.0 context runs
show the same ordering (seed 1 +0.217 vs seed 42 +0.492). The map-level
λ does not explain the excursion: seed 1's longer engine run at
m = 2.5 is the repair ratchet (residual stalling just under target so
steps keep being accepted), an engine-level acceptance phenomenon, not
a more sensitive map. The λ_mean series agrees (seed 1 −0.19, seed 42
−0.18, seed 7 +0.05 — no seed-1 excess).

### (iii) Half-dt decision — DISAGREE on all three seeds: projection/discretisation artifact

λ_max, same 1.25-time-unit window (horizon in brackets):

| seed | dt = 0.05 | dt = 0.025 | gap | verdict |
|------|-----------|------------|-----|---------|
| 42 | +0.499 (R² 0.89) [27.7] | +1.360 (R² 1.00) [10.2] | +0.86 | disagree |
| 1 | +0.247 (R² 0.80) [55.9] | +0.538 (R² 0.93) [25.7] | +0.29 | disagree |
| 7 | −0.446 (R² 0.87) | +0.486 (R² 0.44) [28.4] | +0.93, sign flip | disagree |

The pre-registered tolerance was |Δλ| ≤ 0.15 with the same sign. Every
seed exceeds it; seed 7 flips sign outright (−0.45 → +0.49). The mean
series agrees in direction (seed 42 flips −0.16 → +0.25; seed 7 goes
−0.95 → −0.38). The same-step-count sensitivity (25 steps at both dt)
agrees too: dt/2 gives +1.29 / +0.58 / +1.22 vs +0.50 / +0.25 / −0.45
at dt. **Decision: the half-dt behaviour is a projection /
discretisation artifact, not real dynamics.** Halving dt weakens the
dephasing contraction (~4× per halving) faster than the Euler leak
(~1.4×), so the balance tips toward divergence — the round-6 stage
decomposition predicts exactly this direction. The dt ≈ 0.04 peak in
the Euler+clip+fidelity composition and the λ(dt/2) > λ(dt) growth
are two faces of the same discretisation dependence: λ here is a
property of the discrete map at a given dt, not of a continuous-time
limit (and the dt → 0 limit was not measured — the map at dt/2 is
further from, not closer to, the λ(dt) value).

## Honest limits

- Finite-time λ on a 3×3 lattice of 32-dim cells under a discrete map
  that includes an Euler step, a clip projection (`project_valid`) and
  a nonlinear mirror stage, saturating at trace distance 1.0. It
  measures sensitive dependence of this classical emulation, not chaos
  in any physical system, and says nothing about consciousness.
- λ is a property of the bare map (`propose_step`, no guards, no
  adaptive retry, no calibrator signal), not of the guarded engine.
  Engine-level consequences (breaker latching, repair accepts) are a
  separate question; claim (ii) explicitly tests whether a map-level
  λ explains an engine-level excursion, and the pilot says it does not.
- The blended-fidelity regime is transient: even with matched clocks,
  mean fidelity drifts (below-threshold runs cross 0.90 within 8–25
  steps as dephasing contraction re-synchronises the lattice), so the
  25-step window is a finite-time window on a drifting regime, and the
  horizon t_λ is a formal extrapolation — saturation is never reached
  inside the regime (d grows from 1e-6 to ~3e-6 in the window).
- The 1 − fidelity check series is second order in δ₀ and sits at its
  numerical floor (~1e-11) on contracting runs; it can only corroborate
  diverging runs, where it rises above the floor.
- λ_max (worst cell) and λ_mean (bulk) disagree in sign: the bulk
  contracts while the worst cell's difference grows. Both are
  reported; the primary is the max, the standard worst-case reading of
  sensitive dependence.
- One machine, deterministic seeds; no statistical error bars beyond
  the three-seed replication and the reported R² values.
