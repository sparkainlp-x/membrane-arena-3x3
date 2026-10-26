"""Negative controls: the injection never loosens a guard (SYNTHETIC)."""

import numpy as np
import pytest

from experiments.regimes import REGIMES
from membrane_arena_3x3.core import is_valid_density_matrix
from membrane_arena_3x3.crystallize_injection import InjectionConfig, MembraneArena3x3WithInjection
from membrane_arena_3x3.engine import MembraneArena3x3
from membrane_arena_3x3.shields import YangObserverShield, YinCalibratorShield


def _public(obj):
    return {k: v for k, v in obj.__dict__.items() if k != "log"}


@pytest.mark.parametrize("regime,adaptive", [("stock", True), ("evolved", True), ("intermediate", False)])
def test_every_committed_state_passes_a_fresh_observer(regime, adaptive):
    reg = REGIMES[regime]
    eng = MembraneArena3x3WithInjection(
        reg.engine_config(11),
        InjectionConfig(first_factor=reg.first_factor, mirror_variant=reg.mirror_variant, adaptive=adaptive),
    )
    fresh = YangObserverShield()
    for _ in range(30):
        before = eng.lattice.state.copy()
        result = eng.step()
        if result["status"] == "circuit_breaker_open":
            break
        if result["status"] == "accepted":
            obs = fresh.observe(eng.lattice.state, eng.lattice.reference, eng.current_step)
            assert fresh.is_healthy(obs) == (True, "ok")
            assert all(is_valid_density_matrix(eng.lattice.state[i, j]) for i in range(3) for j in range(3))
        else:
            assert np.array_equal(eng.lattice.state, before)  # a rejected step changes nothing


def test_guard_thresholds_are_the_stock_ones_after_an_adaptive_run():
    eng = MembraneArena3x3WithInjection(
        REGIMES["stock"].engine_config(42), InjectionConfig(adaptive=True)
    )
    eng.run()
    stock = MembraneArena3x3(REGIMES["stock"].engine_config(42))
    assert _public(eng.observer) == _public(YangObserverShield())
    assert _public(eng.calibrator) == _public(YinCalibratorShield())
    assert eng.config.failure_limit == stock.config.failure_limit == 5
    assert eng.config.lattice.__dict__ == stock.config.lattice.__dict__


def test_injection_cannot_accept_a_state_the_stock_guards_reject():
    eng = MembraneArena3x3WithInjection(REGIMES["stock"].engine_config(3), InjectionConfig(adaptive=True))
    real_validate = eng._validate
    calls = []

    def spy(candidate):
        ok, why = real_validate(candidate)
        calls.append(ok)
        return ok, why

    eng._validate = spy
    s = eng.run()
    # every accepted step corresponds to exactly one candidate that passed the stock check
    assert sum(calls) == s["accepted_steps"]
