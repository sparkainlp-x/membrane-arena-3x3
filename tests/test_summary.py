"""The summary always has the same keys, including for a zero-step run."""

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3

KEYS = {
    "status",
    "steps_requested",
    "steps_executed",
    "accepted_steps",
    "rejected_steps",
    "final",
    "observations",
    "calibrations",
}


def test_zero_step_run_has_the_full_summary_shape():
    result = MembraneArena3x3(EngineConfig(steps=0)).run()
    assert set(result) == KEYS
    assert result["status"] == "no_steps"
    assert result["steps_executed"] == 0
    assert result["final"] == {}


def test_normal_run_has_the_same_keys():
    assert set(MembraneArena3x3(EngineConfig(steps=2)).run()) == KEYS
