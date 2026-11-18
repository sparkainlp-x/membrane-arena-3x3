"""Per-cell time difference: dt multipliers and phase offsets.

EXPERIMENTAL parameter (LatticeConfig.dt_multipliers / .phase_offsets).
The defaults (all 1.0 / all 0.0) reproduce the stock update exactly; the
cross-check below pins the new parameter to the earlier H-scaling
experiment: in legacy mode, scaling a cell's Hamiltonian by s is the same
unitary update as scaling its per-step time by s, because the Hamiltonian
appears only in the commutator term.
"""

import pytest

from membrane_arena_3x3.core import fidelity
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.lattice import LatticeConfig
from membrane_arena_3x3.shields import GuardConfig

ROW0 = [(0, 0), (0, 1), (0, 2)]


def _set_identical(arena, cells):
    """Give every cell in `cells` the same state, Hamiltonian and reference."""
    rho = arena.lattice.state[cells[0]].copy()
    h = arena.lattice.hamiltonians[cells[0]].copy()
    for i, j in cells:
        arena.lattice.state[i, j] = rho.copy()
        arena.lattice.hamiltonians[i, j] = h.copy()
        arena.lattice.reference[i, j] = rho.copy()


def _fidelity_trace(arena, cells, max_steps=200):
    """Min pairwise fidelity of `cells` after each accepted step."""
    trace = []
    for _ in range(max_steps):
        result = arena.step()
        if result["status"] == "circuit_breaker_open":
            break
        if result["status"] == "accepted":
            rhos = [arena.lattice.state[i, j] for i, j in cells]
            trace.append(
                min(
                    fidelity(rhos[a], rhos[b])
                    for a in range(3)
                    for b in range(a + 1, 3)
                )
            )
    return trace


def test_time_difference_fields_reject_bad_values():
    with pytest.raises(ValueError):
        LatticeConfig(dt_multipliers=(1.0,) * 8)
    with pytest.raises(ValueError):
        LatticeConfig(phase_offsets=(0.0,) * 10)
    with pytest.raises(ValueError):
        LatticeConfig(dt_multipliers=(1.0,) * 8 + (3.5,))
    with pytest.raises(ValueError):
        LatticeConfig(dt_multipliers=(1.0,) * 8 + (-0.1,))
    with pytest.raises(ValueError):
        LatticeConfig(phase_offsets=(0.0,) * 8 + (0.11,))
    with pytest.raises(ValueError):
        LatticeConfig(phase_offsets=(0.0,) * 8 + (-0.01,))


def test_defaults_reproduce_stock_run_exactly():
    default = MembraneArena3x3(EngineConfig(seed=42)).run()
    explicit = MembraneArena3x3(
        EngineConfig(
            seed=42,
            lattice=LatticeConfig(
                dt_multipliers=(1.0,) * 9, phase_offsets=(0.0,) * 9
            ),
        )
    ).run()
    assert explicit["steps_executed"] == default["steps_executed"] == 12
    assert explicit["accepted_steps"] == default["accepted_steps"] == 7
    assert explicit["final"] == default["final"]


def test_dt_multiplier_reproduces_hamiltonian_scaling():
    """Multipliers (1, 1.5, 2) on identical row-0 cells == H x(1, 1.5, 2).

    The pairwise guard is disabled here on purpose: the test deliberately
    constructs three identical cells, which the adopted pairwise guard
    correctly refuses as a collision. The cross-check is about the
    time-difference parameter, not the guard.
    """
    via_h = MembraneArena3x3(
        EngineConfig(seed=42, guards=GuardConfig(min_pairwise_divergence=None))
    )
    _set_identical(via_h, ROW0)
    via_h.lattice.hamiltonians[0, 1] = via_h.lattice.hamiltonians[0, 1] * 1.5
    via_h.lattice.hamiltonians[0, 2] = via_h.lattice.hamiltonians[0, 2] * 2.0
    trace_h = _fidelity_trace(via_h, ROW0)

    via_dt = MembraneArena3x3(
        EngineConfig(
            seed=42,
            guards=GuardConfig(min_pairwise_divergence=None),
            lattice=LatticeConfig(
                dt_multipliers=(1.0, 1.5, 2.0) + (1.0,) * 6
            ),
        )
    )
    _set_identical(via_dt, ROW0)
    trace_dt = _fidelity_trace(via_dt, ROW0)

    assert len(trace_h) == len(trace_dt) == 6
    assert trace_dt == pytest.approx(trace_h, abs=1e-6)
    # And both diverge monotonically from identical, as measured before.
    assert trace_dt[0] == pytest.approx(0.975, abs=0.01)
    assert trace_dt[-1] == pytest.approx(0.717, abs=0.01)


def test_phase_offset_breaks_identical_cells():
    # Pairwise guard disabled: the test deliberately constructs three
    # identical cells (see test_dt_multiplier_reproduces_hamiltonian_scaling).
    arena = MembraneArena3x3(
        EngineConfig(
            seed=42,
            guards=GuardConfig(min_pairwise_divergence=None),
            lattice=LatticeConfig(
                phase_offsets=(0.0, 0.05, 0.05) + (0.0,) * 6
            ),
        )
    )
    _set_identical(arena, ROW0)
    trace = _fidelity_trace(arena, ROW0)
    assert trace, "run accepted no steps"
    assert trace[0] < 1.0  # the offset cells have already moved apart
