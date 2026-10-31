"""Tests for the experiment helpers (SYNTHETIC). The helpers are not part of the package."""

import numpy as np
import pytest

from experiments import rescoring, stats
from experiments.injection_runs import collect_run
from experiments.regimes import REGIMES
from membrane_arena_3x3.crystallize_injection import InjectionConfig, MembraneArena3x3WithInjection


@pytest.mark.parametrize("regime,seed", [("stock", 42), ("stock", 2), ("intermediate", 5), ("evolved", 3)])
@pytest.mark.parametrize(
    "policy,kw",
    [
        ({"kind": "fixed", "p": 0.35}, dict(first_commit=0.35)),
        ({"kind": "fixed", "p": 0.75}, dict(first_commit=0.75)),
        ({"kind": "adaptive", "first": 0.35, "weight": 4}, dict(first_commit=0.35, adaptive=True)),
        ({"kind": "adaptive", "first": 0.75, "weight": 4}, dict(first_commit=0.75, adaptive=True)),
    ],
)
def test_rescoring_equals_the_engine_ledger(regime, seed, policy, kw):
    reg = REGIMES[regime]
    eng = MembraneArena3x3WithInjection(
        reg.engine_config(seed),
        InjectionConfig(first_factor=reg.first_factor, mirror_variant=reg.mirror_variant, **kw),
    )
    s = eng.run()
    events = [[len(e["attempts"]), bool(e["passed"])] for e in eng.injections]
    firsts = rescoring.policy_first_commits(events, policy)
    assert firsts == pytest.approx([e["attempts"][0]["commit_p"] for e in eng.injections])
    assert rescoring.ledger_per_attempt(events, firsts) == pytest.approx(s["injection_score_total"])


def test_the_commit_policy_never_changes_the_dynamics():
    reg = REGIMES["stock"]
    kw = dict(first_factor=reg.first_factor, mirror_variant=reg.mirror_variant)
    a = MembraneArena3x3WithInjection(reg.engine_config(7), InjectionConfig(first_commit=0.2, **kw)).run()
    b = MembraneArena3x3WithInjection(
        reg.engine_config(7), InjectionConfig(first_commit=0.9, adaptive=True, **kw)
    ).run()
    assert a["observations"] == b["observations"]
    assert [e["passed"] for e in a["injections"]] == [e["passed"] for e in b["injections"]]


def test_collect_run_shape():
    r = collect_run("stock", 42)
    # The injected run is unchanged (11 steps, 6 accepted); only the
    # stock run moved, from 8 to 12 steps, because it now retries.
    assert r["steps"] == 11 and r["accepted"] == 6 and r["stock_steps"] == 12
    assert sum(1 for _, p in r["events"] if p) == 3


def test_holm_and_wilson_and_flip():
    out = stats.holm([0.01, 0.04, 0.03])
    assert [o["p_holm"] for o in out] == pytest.approx([0.03, 0.06, 0.06])
    assert [o["reject"] for o in out] == [True, False, False]
    lo, hi = stats.wilson(10, 10)
    assert hi == pytest.approx(1.0) and 0.7 < lo < 1.0
    assert stats.sign_flip_p(np.ones(30), n_flips=2000) < 0.01
    assert stats.sign_flip_p(np.array([1.0, -1.0] * 20), n_flips=2000) > 0.5


def test_ece_matches_a_hand_computation():
    # two forecasts in the same bin: mean 0.3, observed 0.5 -> ECE 0.2
    res = stats.ece_with_ci([[0.3], [0.3]], [[1], [0]], n_boot=50)
    assert res["ece"] == pytest.approx(0.2)
