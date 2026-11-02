"""The circuit breaker opens after repeated failures and then stops all updates."""

import numpy as np

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3


def force_failures(arena, outcomes):
    """Make _validate return the given (accepted, reason) pairs in order."""
    iterator = iter(outcomes)
    arena._validate = lambda candidate: next(iterator)


def test_breaker_opens_at_exactly_the_failure_limit_and_stops_updates():
    arena = MembraneArena3x3(EngineConfig(steps=100, seed=5, failure_limit=3))
    force_failures(arena, [(False, "forced")] * 100)
    state_before = arena.lattice.state.tobytes()

    result = arena.run()

    assert result["status"] == "circuit_breaker_open"
    assert result["steps_executed"] == 3
    assert result["rejected_steps"] == 3
    assert arena.failures == 3 and arena.circuit_breaker

    # Once open, further steps do nothing: no state change, no new log entries.
    logged = (len(arena.observations), len(arena.calibrations), arena.current_step)
    for _ in range(5):
        assert arena.step()["status"] == "circuit_breaker_open"
    assert (len(arena.observations), len(arena.calibrations), arena.current_step) == logged
    assert arena.lattice.state.tobytes() == state_before
    assert arena.run()["steps_executed"] == 3


def test_breaker_stays_closed_below_the_limit():
    # Retry off here so each forced outcome maps 1:1 to a step; the
    # retry layer's own counting is pinned in test_scientific_upgrade.
    arena = MembraneArena3x3(
        EngineConfig(steps=2, seed=5, failure_limit=3, adaptive_retry=False)
    )
    force_failures(arena, [(False, "forced")] * 2)
    result = arena.run()
    assert result["status"] == "completed"
    assert not arena.circuit_breaker and arena.failures == 2


def test_an_accepted_update_resets_the_consecutive_failure_count():
    # Retry off here so each forced outcome maps 1:1 to a step; the
    # retry layer's own counting is pinned in test_scientific_upgrade.
    arena = MembraneArena3x3(
        EngineConfig(steps=7, seed=5, failure_limit=3, adaptive_retry=False)
    )
    force_failures(
        arena,
        [(False, "x"), (False, "x"), (True, "ok"), (False, "x"), (False, "x"), (True, "ok"), (False, "x")],
    )
    result = arena.run()
    assert result["status"] == "completed"
    assert result["accepted_steps"] == 2 and result["rejected_steps"] == 5
    assert arena.failures == 1


def test_accepted_steps_after_forced_acceptance_are_real_commits():
    arena = MembraneArena3x3(EngineConfig(steps=1, seed=5))
    force_failures(arena, [(True, "ok")])
    before = arena.lattice.state.copy()
    arena.run()
    assert not np.array_equal(before, arena.lattice.state)
