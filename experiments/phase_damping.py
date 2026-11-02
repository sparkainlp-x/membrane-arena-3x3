"""True phase damping as an EXPERIMENT (SYNTHETIC). Does not alter the published model.

The published ``apply_dephasing`` multiplies entry (i, j) by exp(-gamma (i + j)),
which also damps the diagonal and is followed by renormalisation. This module
implements, from first principles, off-diagonal-only damping

    rho_ij -> rho_ij * exp(-gamma * |i - j|)

which leaves the diagonal (populations) and the trace unchanged. It is a Schur
product with the Kac-Murdock-Szego kernel exp(-gamma |i - j|), which is positive
semidefinite, so the map keeps a density matrix positive. The experiment swaps
this in for the dissipative term only; guards, thresholds, bounds and the
breaker are the stock ones.

``PhaseDampedArena`` re-classes the lattice created by the stock engine, so
the initial states and Hamiltonians of a seed are identical to the published
model and runs are paired by seed.
"""

from __future__ import annotations

import numpy as np

from membrane_arena_3x3.core import apply_dephasing, crystallization_mirror, project_valid, residual_infinity
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.lattice import MembraneLattice


def off_diagonal_damping(rho: np.ndarray, gamma: float) -> np.ndarray:
    """Damp off-diagonal entries by exp(-gamma |i - j|); diagonal and trace unchanged."""
    if gamma <= 0:
        return rho
    idx = np.arange(rho.shape[0])
    kernel = np.exp(-gamma * np.abs(idx[:, None] - idx[None, :]))
    return rho * kernel


class PhaseDampingLattice(MembraneLattice):
    """Lattice whose dissipative term is true phase damping (experiment)."""

    damping = staticmethod(off_diagonal_damping)

    def propose_step(self, calibrator_signal: float = 0.0, **_overrides) -> np.ndarray:
        # The engine's retry layer passes dt/mirror/coupling overrides;
        # this experiment's damping uses the lattice config's own
        # values, so overrides are accepted and ignored here.
        cfg = self.config
        candidate = np.empty_like(self.state)
        for i in range(3):
            for j in range(3):
                rho = self.state[i, j]
                h = self.hamiltonians[i, j]
                unitary = -1j * cfg.dt * (h @ rho - rho @ h)
                dissipative = self.damping(rho, cfg.dephasing * cfg.dt)
                delta = unitary + (dissipative - rho)
                delta += self.interaction_delta(i, j) * cfg.dt
                if calibrator_signal:
                    delta += calibrator_signal * cfg.dt * (self.reference[i, j] - rho)
                candidate[i, j] = project_valid(rho + delta)
                candidate[i, j] = crystallization_mirror(candidate[i, j], cfg.mirror_strength)
        return candidate


def phase_damped_arena(config: EngineConfig) -> MembraneArena3x3:
    """Stock engine with its lattice swapped to true phase damping (same initial states)."""
    eng = MembraneArena3x3(config)
    eng.lattice.__class__ = PhaseDampingLattice
    return eng


# ---------------------------------------------------------------------------
# Term attribution of residual growth
# ---------------------------------------------------------------------------

TERMS = ("unitary", "dephasing", "interaction", "calibrator")


def term_deltas(lat: MembraneLattice, i: int, j: int, signal: float) -> dict:
    """The four additive contributions to the update of one cell, as in the published step."""
    cfg = lat.config
    rho = lat.state[i, j]
    h = lat.hamiltonians[i, j]
    return {
        "unitary": -1j * cfg.dt * (h @ rho - rho @ h),
        "dephasing": apply_dephasing(rho, cfg.dephasing * cfg.dt) - rho,
        "interaction": lat.interaction_delta(i, j) * cfg.dt,
        "calibrator": signal * cfg.dt * (lat.reference[i, j] - rho) if signal else np.zeros_like(rho),
    }


def candidate_from_terms(lat: MembraneLattice, signal: float, use: dict, mirror: bool = True, deltas=None) -> np.ndarray:
    """Candidate state built from a subset of terms (use[term] bool), then project and mirror."""
    cfg = lat.config
    out = np.empty_like(lat.state)
    for i in range(3):
        for j in range(3):
            d = deltas[(i, j)] if deltas is not None else term_deltas(lat, i, j, signal)
            delta = sum(d[t] for t in TERMS if use[t])
            c = project_valid(lat.state[i, j] + delta)
            out[i, j] = crystallization_mirror(c, cfg.mirror_strength) if mirror else c
    return out


def max_residual(state: np.ndarray, reference: np.ndarray) -> float:
    return max(residual_infinity(state[i, j], reference[i, j]) for i in range(3) for j in range(3))


def mean_purity(state: np.ndarray) -> float:
    return float(np.mean([np.trace(state[i, j] @ state[i, j]).real for i in range(3) for j in range(3)]))


def attribute_step(lat: MembraneLattice, signal: float) -> dict:
    """Residual growth attributed to each term at the current lattice state.

    Method (all measured, none assumed). Let R(.) be the largest residual
    (infinity norm distance from the reference) over the nine cells and
    R0 = R(current state). For each term X:
      alone[X]       = R(candidate built from X only, then projection and mirror) - R0
      alone_net[X]   = alone[X] minus the same quantity with no term at all
                       (the projection and mirror acting on the unchanged state)
      leave_out[X]   = R(full candidate) - R(candidate without X)
    and for the mirror:
      mirror[full]   = R(full candidate) - R(full candidate with the mirror off)
    Terms interact through the projection and the mirror, so the alone and
    leave-out numbers need not add up; both are reported. Purity change of
    the mean cell is reported for the same constructions.
    """
    deltas = {(i, j): term_deltas(lat, i, j, signal) for i in range(3) for j in range(3)}
    all_on = {t: True for t in TERMS}
    full = candidate_from_terms(lat, signal, all_on, True, deltas)
    r0 = max_residual(lat.state, lat.reference)
    p0 = mean_purity(lat.state)
    r_full = max_residual(full, lat.reference)
    out = {"R0": r0, "growth_full": r_full - r0, "dpurity_full": mean_purity(full) - p0}
    for t in TERMS:
        only = {k: (k == t) for k in TERMS}
        alone = candidate_from_terms(lat, signal, only, True, deltas)
        without = candidate_from_terms(lat, signal, {k: (k != t) for k in TERMS}, True, deltas)
        out[f"alone_{t}"] = max_residual(alone, lat.reference) - r0
        out[f"leave_out_{t}"] = r_full - max_residual(without, lat.reference)
        out[f"dpurity_alone_{t}"] = mean_purity(alone) - p0
    # No term at all: only projection and mirror act on the unchanged state.
    null = candidate_from_terms(lat, signal, {t: False for t in TERMS}, True, deltas)
    out["null_mirror_only"] = max_residual(null, lat.reference) - r0
    for t in TERMS:
        out[f"alone_net_{t}"] = out[f"alone_{t}"] - out["null_mirror_only"]
    nomirror = candidate_from_terms(lat, signal, all_on, False, deltas)
    out["mirror_effect"] = r_full - max_residual(nomirror, lat.reference)
    out["dpurity_mirror"] = mean_purity(full) - mean_purity(nomirror)
    return out
