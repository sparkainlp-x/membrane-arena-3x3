"""Tests for the Crystallize injection engine (membrane-arena-3x3).

SYNTHETIC. Run from the repository root:  pytest tests/test_crystallize_injection.py -q
"""

import numpy as np
import pytest

from membrane_arena_3x3.crystallize_injection import (
    InjectionConfig,
    MembraneArena3x3WithInjection,
    brier,
    calibration_points,
    clamp_p,
    revise_toward_humility,
)
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3


def test_brier_and_calibration_points_match_crystallize():
    assert brier(0.75, True) == pytest.approx(0.0625)
    assert calibration_points(0.75, True) == pytest.approx(75.0)
    assert calibration_points(0.5, True) == pytest.approx(0.0)
    assert calibration_points(0.99, False) < 0  # confidently wrong goes negative
    assert clamp_p(2.0) == 0.99 and clamp_p(-1.0) == 0.01


def test_revision_moves_halfway_to_humility():
    assert revise_toward_humility(0.75) == pytest.approx(0.625)
    assert revise_toward_humility(0.01) == pytest.approx(0.255)


def test_injection_extends_default_runs_without_changing_guards():
    # With the stock engine's adaptive retry OFF, the injection still
    # extends the run exactly as originally measured (8 -> 11 steps).
    base = MembraneArena3x3(
        EngineConfig(steps=200, seed=42, adaptive_retry=False)
    ).run()
    inj = MembraneArena3x3WithInjection(
        EngineConfig(steps=200, seed=42, adaptive_retry=False)
    ).run()
    assert base["steps_executed"] == 8
    assert inj["steps_executed"] > base["steps_executed"]
    assert inj["accepted_steps"] > base["accepted_steps"]
    assert inj["injection_passes"] >= 1
    # The breaker still latches eventually: the injection measures the
    # structural problem, it does not cure it.
    assert inj["status"] == "circuit_breaker_open"


def test_stock_adaptive_retry_subsumes_the_injection_on_seed_42():
    # Measured, seed 42, 200 steps, retry ON (the default): the stock
    # run reaches 12 steps / 7 accepted and the injected run 11 / 6.
    # The injection no longer extends the stock run, because the
    # stock retry now does the same repair work the injection did.
    # This is a measured change in the injection's value, not a
    # failure of either engine: both still latch the breaker.
    base = MembraneArena3x3(EngineConfig(steps=200, seed=42)).run()
    inj = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42)).run()
    assert base["steps_executed"] == 12 and base["accepted_steps"] == 7
    assert inj["steps_executed"] == 11 and inj["accepted_steps"] == 6
    assert inj["status"] == "circuit_breaker_open"


def test_zero_attempts_behaves_exactly_like_stock():
    stock = MembraneArena3x3(EngineConfig(steps=200, seed=42)).run()
    inj = MembraneArena3x3WithInjection(
        EngineConfig(steps=200, seed=42), InjectionConfig(max_attempts=0)
    ).run()
    assert inj["steps_executed"] == stock["steps_executed"]
    assert inj["accepted_steps"] == stock["accepted_steps"]
    assert inj["injections"] == []


def test_injected_runs_are_deterministic():
    a = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42)).run()
    b = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42)).run()
    assert a["injections"] == b["injections"]
    assert a["observations"] == b["observations"]


def test_counts_match_stock_layout():
    eng = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42))
    summary = eng.run()
    assert len(eng.observations) == summary["steps_executed"]
    assert len(eng.calibrations) == summary["steps_executed"]
    assert summary["injection_events"] == len(summary["injections"])


def test_variant_proposal_restores_dt_and_mirror():
    eng = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42))
    dt0 = eng.lattice.config.dt
    mirror0 = eng.lattice.config.mirror_strength
    eng._propose_with_dt_factor(0.05, 0.5)
    assert eng.lattice.config.dt == dt0
    assert eng.lattice.config.mirror_strength == mirror0


def test_failed_variants_change_nothing():
    eng = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=0))
    # Step until the first rejection, then check the injection attempts on
    # that step left the state untouched when they failed.
    for _ in range(20):
        before = eng.lattice.state.copy()
        result = eng.step()
        if result["status"] == "rejected" and not (
            eng.injections and eng.injections[-1]["passed"]
        ):
            assert np.array_equal(eng.lattice.state, before)
            break
    else:
        pytest.skip("no clean failed injection observed in 20 steps")


def test_variant_factors_always_shrink_the_failed_step():
    # Even with first_factor 0.9, jitter must never push a variant above 1.
    eng = MembraneArena3x3WithInjection(
        EngineConfig(steps=200, seed=42),
        InjectionConfig(first_factor=0.9),
    )
    eng.run()
    for event in eng.injections:
        for attempt in event["attempts"]:
            assert attempt["factor"] < 1.0


def test_invalid_injection_config_is_rejected():
    with pytest.raises(ValueError):
        InjectionConfig(max_attempts=-1)
    with pytest.raises(ValueError):
        InjectionConfig(first_factor=1.5)
    with pytest.raises(ValueError):
        InjectionConfig(first_commit=0.0)


def test_adaptive_commit_moves_toward_the_measured_rate():
    # Stock regime, starting overconfident at 0.75: adaptive commits must
    # drift down toward the ~0.31 empirical rate instead of staying put.
    eng = MembraneArena3x3WithInjection(
        EngineConfig(steps=200, seed=42),
        InjectionConfig(first_commit=0.75, adaptive=True),
    )
    eng.run()
    commits = [e["attempts"][0]["commit_p"] for e in eng.injections]
    assert len(commits) >= 2
    assert commits[-1] < commits[0]
    rate = sum(1 for e in eng.injections if e["passed"]) / len(eng.injections)
    assert abs(commits[-1] - rate) < abs(commits[0] - rate)


def test_adaptive_beats_fixed_when_the_regime_moves():
    # Evolved genome (pass rate ~0.93): fixed 0.35 is underconfident there,
    # adaptive must score higher on the same genome and seed.
    from membrane_arena_3x3.lattice import LatticeConfig

    cfg = EngineConfig(
        steps=200,
        seed=42,
        lattice=LatticeConfig(
            coupling=0.0339, dephasing=0.2871, mirror_strength=0.1725, dt=0.0128
        ),
    )
    inj = InjectionConfig(first_factor=0.2555, mirror_variant=0.0)
    fixed = MembraneArena3x3WithInjection(
        cfg, InjectionConfig(first_factor=0.2555, mirror_variant=0.0)
    ).run()
    adaptive = MembraneArena3x3WithInjection(
        cfg,
        InjectionConfig(
            first_factor=0.2555, mirror_variant=0.0, adaptive=True
        ),
    ).run()
    assert adaptive["injection_score_total"] > fixed["injection_score_total"]
    assert adaptive["adaptive_pass_rate"] > 0.8


def test_adaptive_stays_deterministic():
    a = MembraneArena3x3WithInjection(
        EngineConfig(steps=200, seed=42),
        InjectionConfig(adaptive=True),
    ).run()
    b = MembraneArena3x3WithInjection(
        EngineConfig(steps=200, seed=42),
        InjectionConfig(adaptive=True),
    ).run()
    assert a["injections"] == b["injections"]


def test_scores_punish_overconfidence_and_reward_calibration():
    # Explicit 0.75 against a ~0.31 truth must score below 0: the ledger
    # punishes overconfidence instead of hiding it.
    over = MembraneArena3x3WithInjection(
        EngineConfig(steps=200, seed=42), InjectionConfig(first_commit=0.75)
    )
    over.run()
    assert over.injection_score_total < 0
    # The measured default (0.35, near the 0.31 empirical rate) scores
    # above zero across seeds 42, 0-3 with identical dynamics: p affects
    # only the ledger, never acceptance.
    total = 0.0
    for seed in (42, 0, 1, 2, 3):
        eng = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=seed))
        eng.run()
        total += eng.injection_score_total
        for event in eng.injections:
            assert event["sampling_cost"] == pytest.approx(
                0.8 * len(event["attempts"])
            )
    assert total > 0
