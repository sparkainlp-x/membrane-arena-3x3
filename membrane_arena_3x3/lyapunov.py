# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
"""Finite-time Lyapunov exponent for the synthetic membrane lattice.

Twin-run measurement of sensitive dependence of the lattice MAP
(``MembraneLattice.propose_step`` with calibrator signal 0), NOT of the
guarded engine and NOT of ``shields.Observation.lyapunov_proxy``: that
name is a kept alias for ``stability_penalty``, a weighted penalty whose
own docstring says it is not a Lyapunov function.

Method (see ``REPORT.md`` for the pre-registration):
- Build two identical lattices from the same seed. Perturb one cell of
  the second twin by ``delta0`` (default 1e-6) in a trace-preserving way:
  ``rho' = (1 - eps) * rho + eps * sigma`` with ``eps`` chosen so the
  trace distance between ``rho`` and ``rho'`` is exactly ``delta0``.
- Step both twins with the same map and track the distance between
  them. PRIMARY distance: maximum over the nine cells of the trace
  distance (0.5 * ||rho - sigma||_1). CHECK distance: maximum over the
  nine cells of ``1 - fidelity``.
- Finite-time exponent ``lambda``: least-squares slope of ``log d(t)``
  against time ``t = step * dt`` over the exponential-growth window
  (distances from ``2 * delta0`` up to 0.25 of saturation, widened to
  0.5 / 0.9 only if fewer than 5 points fall in the window; if the
  distance never leaves the neighbourhood of ``delta0`` the fit uses
  all points). Horizon ``t_lambda = (1 / lambda) * ln(d_sat / delta0)``
  for ``lambda > 0``, with ``d_sat = 1.0`` (the maximum trace distance
  between two density matrices).

Scope, stated plainly: this measures sensitive dependence of a
classical, discrete, 32-dimensional-per-cell map that includes an
Euler/clip projection stage and a mirror stage, saturating at trace
distance 1. It is a finite-time exponent on a 3x3 lattice, not chaos in
any physical system, and it says nothing about consciousness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import fidelity, project_valid, random_density_matrix
from .lattice import LatticeConfig, MembraneLattice

DEFAULT_DELTA0 = 1e-6
# Maximum trace distance between two density matrices: saturation level.
D_SAT = 1.0


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return 0.5 * ||rho - sigma||_1 (trace norm), in [0, 1]."""
    diff = (rho - sigma + (rho - sigma).conj().T) / 2
    vals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(vals)))


def twin_trace_distance(
    state_a: np.ndarray, state_b: np.ndarray
) -> float:
    """Return the maximum trace distance over the nine cells."""
    return max(
        trace_distance(state_a[i, j], state_b[i, j])
        for i in range(3)
        for j in range(3)
    )


def mean_twin_trace_distance(
    state_a: np.ndarray, state_b: np.ndarray
) -> float:
    """Return the mean trace distance over the nine cells.

    The maximum is the primary series (worst-case spread); the mean
    is recorded alongside it because the maximum of nine noisy
    per-cell trajectories is itself noisy, while the mean averages
    the per-cell fluctuations out.
    """
    total = 0.0
    for i in range(3):
        for j in range(3):
            total += trace_distance(state_a[i, j], state_b[i, j])
    return total / 9.0


def twin_infidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Return the maximum ``1 - fidelity`` over the nine cells (check)."""
    return max(
        1.0 - fidelity(state_a[i, j], state_b[i, j])
        for i in range(3)
        for j in range(3)
    )


def mean_pairwise_fidelity(state: np.ndarray) -> float:
    """Return the mean fidelity over the 36 unordered cell pairs."""
    total = 0.0
    count = 0
    cells = [(i, j) for i in range(3) for j in range(3)]
    for a in range(9):
        for b in range(a + 1, 9):
            total += fidelity(state[cells[a]], state[cells[b]])
            count += 1
    return total / count


def perturb_toward(
    rho: np.ndarray, sigma: np.ndarray, delta0: float = DEFAULT_DELTA0
) -> np.ndarray:
    """Return ``(1 - eps) * rho + eps * sigma`` at trace distance ``delta0``.

    The blend of two density matrices is a density matrix, so the
    perturbation is exactly trace-preserving and PSD by construction
    (no projection/clipping is involved). ``eps = delta0 / d(rho,
    sigma)``; raises ``ValueError`` if the two states are already
    closer than ``delta0``.
    """
    d = trace_distance(rho, sigma)
    if not d > delta0:
        raise ValueError(
            f"perturbation target {delta0:g} exceeds the distance "
            f"to the blend target ({d:g}); pick a more distant sigma"
        )
    eps = delta0 / d
    return (1.0 - eps) * rho + eps * sigma


def blend_to_mean_fidelity(
    lattice: MembraneLattice, target: float, iterations: int = 40
) -> float:
    """Blend every cell toward cell (0, 0) until the mean pairwise fidelity hits ``target``.

    Bisection on the blend factor ``a`` in [0, 1]: ``state[k] =
    (1 - a) * state[k] + a * state[0, 0]``. The mean pairwise fidelity
    is monotone in ``a``. Returns the achieved mean pairwise fidelity.
    Hamiltonians are left untouched (distinct clocks keep diverging the
    cells); only the states are blended.
    """
    anchor = lattice.state[0, 0].copy()
    original = lattice.state.copy()
    lo, hi = 0.0, 1.0
    achieved = mean_pairwise_fidelity(lattice.state)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        candidate = np.empty_like(original)
        for i in range(3):
            for j in range(3):
                candidate[i, j] = project_valid(
                    (1.0 - mid) * original[i, j] + mid * anchor
                )
        achieved = mean_pairwise_fidelity(candidate)
        if achieved < target:
            lo = mid
        else:
            hi = mid
    final = np.empty_like(original)
    for i in range(3):
        for j in range(3):
            final[i, j] = project_valid(
                (1.0 - hi) * original[i, j] + hi * anchor
            )
    lattice.state = final
    return mean_pairwise_fidelity(lattice.state)


def _blend_pair_to_fidelity(
    rho0: np.ndarray, rho1: np.ndarray, target: float
) -> np.ndarray:
    """Blend ``rho1`` toward ``rho0`` until their fidelity is ``target``."""
    lo, hi = 0.0, 1.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if fidelity(rho0, project_valid((1.0 - mid) * rho1 + mid * rho0)) < target:
            lo = mid
        else:
            hi = mid
    return project_valid((1.0 - hi) * rho1 + hi * rho0)


def setup_matched_hamiltonians(lattice: MembraneLattice) -> None:
    """Give every cell cell (0, 0)'s Hamiltonian (matched clocks).

    With distinct Hamiltonians the unitary stage rotates the cells
    apart in a single step, so a lattice blended to mean fidelity above
    0.90 collapses below 0.90 after one step and the above-threshold
    regime cannot be measured. Matching the clocks removes that
    one-step blow-apart (an exact unitary with a shared H changes pair
    fidelity by exactly 0; the Euler stage leaves only its small
    discretisation error), so the blended fidelity persists and the
    interaction term -- the mechanism under test -- is what moves it.
    """
    h = lattice.hamiltonians[0, 0].copy()
    for i in range(3):
        for j in range(3):
            lattice.hamiltonians[i, j] = h.copy()


def setup_trio(
    lattice: MembraneLattice,
    cells: list[tuple[int, int]],
    start: float = 0.85,
) -> None:
    """Re-pin a three-cell subgroup to the round-6 re-pinned construction.

    Cells 1 and 2 are blended toward cell 0 until each pair fidelity is
    ``start``; all three cells then share cell 0's Hamiltonian, and the
    references of cells 1 and 2 are set to cell 0's state. This is the
    construction behind the seed-1 excursion (``scratch-round6/
    exp_seed1.py``); the pairwise guard would refuse an exactly
    identical trio at step 0, so the re-pinned 0.85 start is the
    comparison construction.
    """
    rho0 = lattice.state[cells[0]].copy()
    for other in cells[1:]:
        lattice.state[other] = _blend_pair_to_fidelity(
            rho0, lattice.state[other], start
        )
    h = lattice.hamiltonians[cells[0]].copy()
    for i, j in cells:
        lattice.hamiltonians[i, j] = h.copy()
    lattice.reference[cells[1]] = lattice.state[cells[0]].copy()
    lattice.reference[cells[2]] = lattice.state[cells[0]].copy()


@dataclass
class LambdaFit:
    """Result of the log-linear fit of ``log d(t)`` against ``t``."""

    lam: float | None
    intercept: float
    r_squared: float
    n_points: int
    window_start: float
    window_end: float
    window_upper: float
    horizon: float | None  # (1/lam) * ln(d_sat/delta0) if lam > 0 else None
    fitted: bool  # False when the series never rose above its noise floor


def fit_lambda(
    times: np.ndarray,
    distances: np.ndarray,
    delta0: float = DEFAULT_DELTA0,
    d_sat: float = D_SAT,
    min_points: int = 5,
    noise_floor: float = 1e-9,
    window_steps: int | None = None,
) -> LambdaFit:
    """Fit the finite-time exponent as the slope of ``log d`` vs ``t``.

    The fit excludes t = 0 (anchored by construction at ``delta0``).
    With ``window_steps`` given (the pre-registered protocol: first 25
    steps, 1.25 time units at dt = 0.05), the fit uses exactly steps
    1..``window_steps``. Without it, the window is distance-based:
    points with ``2 * delta0 <= d <= upper``, with ``upper`` widened
    0.25 -> 0.5 -> 0.9 until at least ``min_points`` fall in the
    window; if the distance never reaches ``2 * delta0`` (contracting
    or neutral run) the fit uses all t > 0 points. Distances are floored
    at 1e-15 before the logarithm. If the series never rises above
    ``noise_floor`` (e.g. ``1 - fidelity`` for a ``delta0 = 1e-6``
    perturbation: it is second order in the perturbation, so it sits at
    ~1e-12, the numerical floor of the fidelity computation), no fit is
    attempted and ``fitted`` is False.
    """
    times = np.asarray(times, dtype=float)
    distances = np.asarray(distances, dtype=float)
    moving = times > 0
    t = times[moving]
    d = np.maximum(distances[moving], 1e-15)
    if not d.size or float(np.max(d)) <= noise_floor:
        return LambdaFit(
            lam=None,
            intercept=float("nan"),
            r_squared=float("nan"),
            n_points=0,
            window_start=float(t[0]) if t.size else 0.0,
            window_end=float(t[-1]) if t.size else 0.0,
            window_upper=float("nan"),
            horizon=None,
            fitted=False,
        )
    if window_steps is not None:
        n = min(window_steps, d.size)
        window = np.zeros_like(d, dtype=bool)
        window[:n] = True
        window_upper = float(np.max(d[:n]))
    else:
        window_upper = 0.9
        window = np.ones_like(d, dtype=bool)
        for upper in (0.25, 0.5, 0.9):
            candidate = (d >= 2.0 * delta0) & (d <= upper)
            if candidate.sum() >= min_points or upper == 0.9:
                window = candidate
                window_upper = upper
                break
        if window.sum() < 3:
            window = np.ones_like(d, dtype=bool)
            window_upper = float(np.max(d)) if d.size else 0.9
    tw = t[window]
    lw = np.log(d[window])
    slope, intercept = np.polyfit(tw, lw, 1)
    predicted = slope * tw + intercept
    ss_res = float(np.sum((lw - predicted) ** 2))
    ss_tot = float(np.sum((lw - np.mean(lw)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    horizon = (
        float(np.log(d_sat / delta0) / slope) if slope > 0 else None
    )
    return LambdaFit(
        lam=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        n_points=int(window.sum()),
        window_start=float(tw[0]),
        window_end=float(tw[-1]),
        window_upper=float(window_upper),
        horizon=horizon,
        fitted=True,
    )


@dataclass
class TwinRun:
    """One twin run: trajectories plus the fitted exponent."""

    seed: int
    steps_executed: int
    times: list[float]
    trace_distances: list[float]
    mean_trace_distances: list[float]
    infidelities: list[float]
    mean_pairwise_fidelity: list[float]
    initial_mean_fidelity: float
    saturated: bool
    collapsed: bool
    fit: LambdaFit
    fit_check: LambdaFit  # same fit on the 1 - fidelity check distance

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "steps_executed": self.steps_executed,
            "times": self.times,
            "trace_distances": self.trace_distances,
            "mean_trace_distances": self.mean_trace_distances,
            "infidelities": self.infidelities,
            "mean_pairwise_fidelity": self.mean_pairwise_fidelity,
            "initial_mean_fidelity": self.initial_mean_fidelity,
            "saturated": self.saturated,
            "collapsed": self.collapsed,
            "lambda": self.fit.lam,
            "lambda_fitted": self.fit.fitted,
            "lambda_intercept": self.fit.intercept,
            "lambda_r_squared": self.fit.r_squared,
            "lambda_n_points": self.fit.n_points,
            "lambda_window": [self.fit.window_start, self.fit.window_end],
            "lambda_window_upper": self.fit.window_upper,
            "horizon": self.fit.horizon,
            "lambda_check_infidelity": self.fit_check.lam,
            "lambda_check_fitted": self.fit_check.fitted,
            "lambda_check_r_squared": self.fit_check.r_squared,
            "lambda_check_n_points": self.fit_check.n_points,
            "horizon_check": self.fit_check.horizon,
        }


def twin_run(
    config: LatticeConfig,
    seed: int,
    steps: int,
    delta0: float = DEFAULT_DELTA0,
    perturb_cells: tuple[tuple[int, int], ...] | None = None,
    setup=None,
    perturb_seed: int = 20260717,
    stop_at_saturation: float = 0.9,
    stop_at_collapse: float = 1e-12,
    fit_window_steps: int | None = None,
) -> TwinRun:
    """Run twin lattices and fit the finite-time Lyapunov exponent.

    Both twins are built from ``seed`` (identical state and
    Hamiltonians). ``setup`` (if given) is applied identically to both
    twins before the perturbation; it must be deterministic. Every cell
    (or every cell in ``perturb_cells``) of the second twin is then
    perturbed by ``delta0`` (trace distance) toward an independent
    random density matrix drawn from ``perturb_seed``.

    All-cell perturbation is deliberate: perturbing a single cell
    measures mostly that cell's own contraction (its neighbours enter
    its update only through the O(coupling * dt) interaction, so the
    perturbed cell's distance decays at the single-cell rate whatever
    the lattice regime is). Perturbing the whole state vector lets the
    interaction term -- whose sign flips at the 0.90 fidelity
    threshold -- act on the difference field itself, which is what
    claim (i) is about.

    Both twins are stepped with the bare map (``propose_step(0.0)`` +
    ``commit``): no guards, no adaptive retry, no calibrator signal,
    so ``lambda`` measures the map itself, not guard/retry branching.
    Stops early at saturation (max trace distance >= 0.9) or collapse
    (<= 1e-12).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0.0 < delta0 < 1.0:
        raise ValueError("delta0 must be in (0, 1)")
    lat_a = MembraneLattice(config, np.random.default_rng(seed))
    lat_b = MembraneLattice(config, np.random.default_rng(seed))
    if setup is not None:
        setup(lat_a)
        setup(lat_b)
    cells = perturb_cells or tuple(
        (i, j) for i in range(3) for j in range(3)
    )
    rng = np.random.default_rng(perturb_seed)
    for cell in cells:
        sigma = random_density_matrix(rng)
        rho = lat_b.state[cell].copy()
        lat_b.state[cell] = perturb_toward(rho, sigma, delta0)

    times = [0.0]
    trace_distances = [twin_trace_distance(lat_a.state, lat_b.state)]
    mean_trace_distances = [
        mean_twin_trace_distance(lat_a.state, lat_b.state)
    ]
    infidelities = [twin_infidelity(lat_a.state, lat_b.state)]
    mean_fid = [mean_pairwise_fidelity(lat_a.state)]
    saturated = False
    collapsed = False
    dt = config.dt
    for step in range(1, steps + 1):
        cand_a = lat_a.propose_step(0.0)
        cand_b = lat_b.propose_step(0.0)
        lat_a.commit(cand_a)
        lat_b.commit(cand_b)
        times.append(step * dt)
        trace_distances.append(twin_trace_distance(lat_a.state, lat_b.state))
        mean_trace_distances.append(
            mean_twin_trace_distance(lat_a.state, lat_b.state)
        )
        infidelities.append(twin_infidelity(lat_a.state, lat_b.state))
        mean_fid.append(mean_pairwise_fidelity(lat_a.state))
        if trace_distances[-1] >= stop_at_saturation:
            saturated = True
            break
        if trace_distances[-1] <= stop_at_collapse:
            collapsed = True
            break
    fit = fit_lambda(
        np.asarray(times),
        np.asarray(trace_distances),
        delta0,
        window_steps=fit_window_steps,
    )
    fit_check = fit_lambda(
        np.asarray(times),
        np.asarray(infidelities),
        delta0,
        window_steps=fit_window_steps,
    )
    return TwinRun(
        seed=seed,
        steps_executed=len(times) - 1,
        times=times,
        trace_distances=trace_distances,
        mean_trace_distances=mean_trace_distances,
        infidelities=infidelities,
        mean_pairwise_fidelity=mean_fid,
        initial_mean_fidelity=mean_fid[0],
        saturated=saturated,
        collapsed=collapsed,
        fit=fit,
        fit_check=fit_check,
    )


def measure_lambda(
    config: LatticeConfig,
    seed: int,
    steps: int,
    **kwargs,
) -> dict:
    """Run one twin measurement and return its JSON-able summary."""
    run = twin_run(config, seed, steps, **kwargs)
    return run.to_dict()
