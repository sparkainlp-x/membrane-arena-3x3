"""A rejected update must leave the whole lattice bit-identical (all or nothing)."""

import numpy as np

from membrane_arena_3x3.core import is_valid_density_matrix
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.lattice import LatticeConfig


def snapshot(arena):
    lat = arena.lattice
    return (
        lat.state.tobytes(),
        lat.reference.tobytes(),
        lat.hamiltonians.tobytes(),
    )


def partly_bad_candidate(arena):
    """Cell [0,0] changed but valid; cell [2,2] invalid (negative eigenvalue)."""
    candidate = arena.lattice.state.copy()
    candidate[0, 0] = np.eye(arena.lattice.config.dim) / arena.lattice.config.dim
    bad = candidate[2, 2].copy()
    bad[0, 0] -= 1.0
    candidate[2, 2] = bad
    assert is_valid_density_matrix(candidate[0, 0])
    assert not is_valid_density_matrix(candidate[2, 2])
    return candidate


def test_one_invalid_cell_rejects_the_entire_update(monkeypatch):
    arena = MembraneArena3x3(EngineConfig(seed=8))
    monkeypatch.setattr(arena.lattice, "propose_step", lambda signal=0.0: partly_bad_candidate(arena))
    before = snapshot(arena)

    result = arena.step()

    assert result["status"] == "rejected"
    assert result["reason"] == "invalid_density_matrix[2,2]"
    assert snapshot(arena) == before  # cell [0,0] was not applied either
    assert arena.failures == 1
    assert arena.calibrations[-1]["accepted"] is False


def test_each_guard_rejects_without_touching_the_state(monkeypatch):
    for attribute, value, expected in [
        ("residual_target", -1.0, "residual_above_target"),
        ("divergence_max", -1.0, "divergence_above_maximum"),
    ]:
        arena = MembraneArena3x3(EngineConfig(seed=8))
        setattr(arena.observer, attribute, value)
        before = snapshot(arena)
        result = arena.step()
        assert (result["status"], result["reason"]) == ("rejected", expected)
        assert snapshot(arena) == before

    arena = MembraneArena3x3(EngineConfig(seed=8))
    monkeypatch.setattr("membrane_arena_3x3.shields.purity", lambda rho: 0.0)
    before = snapshot(arena)
    assert arena.step()["reason"] == "purity_too_low"
    assert snapshot(arena) == before


def test_out_of_bounds_parameter_rejects_every_step_and_never_changes_state():
    lattice = LatticeConfig(coupling=0.5)  # above coupling_max = 0.10
    arena = MembraneArena3x3(EngineConfig(lattice=lattice, steps=50, failure_limit=3))
    before = snapshot(arena)

    result = arena.run()

    assert {c["reason"] for c in result["calibrations"]} == {
        "calibrator_parameter_out_of_bounds"
    }
    assert result["accepted_steps"] == 0
    assert result["steps_executed"] == 3
    assert snapshot(arena) == before


def test_an_accepted_update_changes_the_state_and_keeps_it_valid():
    arena = MembraneArena3x3(EngineConfig(seed=11))
    arena.observer.residual_target = 10.0  # make sure the first step is accepted
    arena.observer.divergence_max = 10.0
    before = arena.lattice.state.copy()
    assert arena.step()["status"] == "accepted"
    assert not np.array_equal(before, arena.lattice.state)
    for i in range(3):
        for j in range(3):
            rho = arena.lattice.state[i, j]
            assert np.allclose(rho, rho.conj().T, atol=1e-10)
            assert abs(np.trace(rho).real - 1) < 1e-10
            assert np.linalg.eigvalsh(rho).min() > -1e-10
