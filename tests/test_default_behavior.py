"""Pins the measured default behavior documented under "Known limits".

With the default settings the lattice drifts out of the guard envelope within
a few steps, the update is rejected, the state does not change, the retry is
therefore identical, and the breaker latches after five identical rejections.
If the model or the defaults are changed on purpose, update these numbers and
the README together.
"""

import pytest

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("schumann", [False, True])
def test_default_run_latches_the_breaker_within_a_dozen_steps(seed, schumann):
    result = MembraneArena3x3(EngineConfig(seed=seed, schumann=schumann)).run()
    assert result["status"] == "circuit_breaker_open"
    assert 6 <= result["steps_executed"] <= 12
    assert result["rejected_steps"] == 5
    reasons = {c["reason"] for c in result["calibrations"] if not c["accepted"]}
    assert reasons <= {"residual_above_target", "divergence_above_maximum", "purity_too_low"}


def test_retry_after_a_rejection_is_deterministic_without_the_sinusoid():
    arena = MembraneArena3x3(EngineConfig(seed=42))
    arena.run()
    rejected = [c for c in arena.calibrations if not c["accepted"]]
    assert len(rejected) == 5
    assert len({c["reason"] for c in rejected}) == 1
    assert len({c["total_signal"] for c in rejected}) == 1  # identical proposal each time


def test_the_sinusoid_barely_changes_a_default_run():
    plain = MembraneArena3x3(EngineConfig(seed=42)).run()["final"]
    wave = MembraneArena3x3(EngineConfig(seed=42, schumann=True)).run()["final"]
    assert plain["max_residual"] == pytest.approx(wave["max_residual"], abs=1e-3)
    assert plain["min_purity"] == pytest.approx(wave["min_purity"], abs=1e-3)
