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
    base = MembraneArena3x3(EngineConfig(steps=200, seed=42)).run()
    inj = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42)).run()
    assert base["steps_executed"] == 8
    assert inj["steps_executed"] > base["steps_executed"]
    assert inj["accepted_steps"] > base["accepted_steps"]
    assert inj["injection_passes"] >= 1
    # The breaker still latches eventually: the injection measures the
    # structural problem, it does not cure it.
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


def test_scores_are_negative_when_commits_are_overconfident():
    # Default commit 0.75 against a usually-false truth must score below 0:
    # the ledger punishes overconfidence instead of hiding it.
    eng = MembraneArena3x3WithInjection(EngineConfig(steps=200, seed=42))
    eng.run()
    assert eng.injection_score_total < 0
    for event in eng.injections:
        assert event["sampling_cost"] == pytest.approx(
            0.8 * len(event["attempts"])
        )
