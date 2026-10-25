"""Tests for the synthetic 3x3 membrane arena engine."""

import numpy as np

from membrane_arena_3x3.core import is_valid_density_matrix
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3

def test_short_run_produces_valid_summary():
    result = MembraneArena3x3(EngineConfig(steps=5, seed=7)).run()
    assert result["status"] in {"completed", "circuit_breaker_open"}
    assert result["steps_executed"] == 5
    assert result["accepted_steps"] + result["rejected_steps"] == 5

def test_state_remains_valid_after_commit():
    arena = MembraneArena3x3(EngineConfig(steps=3, seed=11))
    arena.run()

    for i in range(3):
        for j in range(3):
            rho = arena.lattice.state[i, j]
            assert is_valid_density_matrix(rho)

def test_observer_does_not_mutate_state():
    arena = MembraneArena3x3(EngineConfig(steps=1, seed=3))
    before = arena.lattice.state.copy()

    arena.observer.observe(
        arena.lattice.state, arena.lattice.reference, 0
    )

    assert np.array_equal(before, arena.lattice.state)

def test_circuit_breaker_limits_failures():
    arena = MembraneArena3x3(
        EngineConfig(steps=20, seed=5, failure_limit=2)
    )
    result = arena.run()

    assert result["status"] in {"completed", "circuit_breaker_open"}
    assert arena.failures <= 2
