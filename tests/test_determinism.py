"""Same seed, same result; different seed, different result; no hidden global state."""

import json

import numpy as np

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3, run_simulation


def dump(result):
    return json.dumps(result, sort_keys=True)


def test_same_seed_gives_identical_results():
    a = run_simulation(EngineConfig(steps=6, seed=123, schumann=True))
    b = run_simulation(EngineConfig(steps=6, seed=123, schumann=True))
    assert dump(a) == dump(b)


def test_different_seeds_give_different_results():
    a = run_simulation(EngineConfig(steps=3, seed=1))
    b = run_simulation(EngineConfig(steps=3, seed=2))
    assert dump(a) != dump(b)


def test_global_numpy_random_state_does_not_matter():
    np.random.seed(0)
    a = run_simulation(EngineConfig(steps=3, seed=9))
    np.random.seed(999)
    np.random.random(100)
    b = run_simulation(EngineConfig(steps=3, seed=9))
    assert dump(a) == dump(b)


def test_interleaved_runs_do_not_interfere():
    first = MembraneArena3x3(EngineConfig(steps=4, seed=3))
    second = MembraneArena3x3(EngineConfig(steps=4, seed=3))
    for _ in range(4):
        first.step()
        second.step()
    assert dump(first.summary()) == dump(second.summary())
