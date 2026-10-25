"""Calling run() again continues the logs instead of restarting the counter."""

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3


def test_second_run_continues_step_numbers():
    arena = MembraneArena3x3(EngineConfig(steps=2, seed=1))
    arena.run()
    result = arena.run()
    steps = [entry["step"] for entry in result["calibrations"]]
    assert steps == [0, 1, 2, 3]
    assert [o["step"] for o in result["observations"]] == [1, 2, 3, 4]
