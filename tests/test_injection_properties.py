"""Extra properties of the opt-in Crystallize injection (SYNTHETIC)."""

import numpy as np
import pytest

from membrane_arena_3x3.crystallize_injection import (
    InjectionConfig,
    MembraneArena3x3WithInjection,
    calibration_points,
    run_simulation_with_injection,
)
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3


def test_injection_never_draws_from_the_main_generator():
    stock = MembraneArena3x3(EngineConfig(steps=200, seed=42))
    stock.run()
    inj = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42))
    inj.run()
    assert inj.injections  # the injection did run
    assert inj.rng.bit_generator.state == stock.rng.bit_generator.state


def test_guards_and_breaker_are_untouched():
    eng = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42))
    before = (
        eng.config.lattice.__dict__.copy(),
        eng.observer.__dict__.copy(),
        eng.config.failure_limit,
    )
    eng.run()
    after = (
        eng.config.lattice.__dict__.copy(),
        eng.observer.__dict__.copy(),
        eng.config.failure_limit,
    )
    assert before == after  # dt, mirror, thresholds all restored/unchanged


def test_ledger_is_consistent():
    eng = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42))
    summary = eng.run()
    total = 0.0
    for event in eng.injections:
        assert 1 <= len(event["attempts"]) <= 3
        # only the last attempt can pass, and it ends the event
        assert [a["passed"] for a in event["attempts"][:-1]] == [False] * (
            len(event["attempts"]) - 1
        )
        assert event["passed"] == event["attempts"][-1]["passed"]
        for a in event["attempts"]:
            assert a["points"] == pytest.approx(calibration_points(a["p"], a["passed"]))
        total += event["score"]
    assert summary["injection_score_total"] == pytest.approx(total)
    assert summary["injection_passes"] == sum(e["passed"] for e in eng.injections)


def test_a_passed_injection_is_logged_as_an_accepted_step():
    eng = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42))
    results = []
    while not eng.circuit_breaker:
        results.append(eng.step())
    injected = [r for r in results if r["injected"]]
    assert injected
    for r in injected:
        assert r["status"] == "accepted"
        assert r["calibration"]["accepted"] is True


def test_injection_cannot_rescue_forced_failures():
    eng = MembraneArena3x3WithInjection(EngineConfig(steps=100, seed=5, failure_limit=3))
    eng._validate = lambda candidate: (False, "forced")
    result = eng.run()
    assert result["status"] == "circuit_breaker_open"
    assert result["steps_executed"] == 3
    assert result["injection_passes"] == 0
    assert all(len(e["attempts"]) == 3 for e in result["injections"])


def test_summary_runner_and_config_validation():
    s = run_simulation_with_injection(EngineConfig(steps=5, seed=1))
    assert s["steps_executed"] == len(s["observations"])
    with pytest.raises(ValueError):
        InjectionConfig(max_attempts=-1)
    with pytest.raises(ValueError):
        InjectionConfig(commit_p=1.0)
