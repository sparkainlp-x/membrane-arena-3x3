"""Tests for scripts/injection_multiseed.py (SYNTHETIC)."""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "injection_multiseed", os.path.join(ROOT, "scripts", "injection_multiseed.py")
)
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def fake_run(seed, stock_acc, inj_acc, events, status="circuit_breaker_open"):
    inj = {"steps": inj_acc + 5, "accepted": inj_acc, "status": status, "engine_score_total": 0.0, "events": events}
    return {
        "seed": seed,
        "stock": {"steps": stock_acc + 5, "accepted": stock_acc, "status": "circuit_breaker_open"},
        "injected": {"default": inj, "dt_only": dict(inj)},
    }


def test_bootstrap_interval_contains_the_mean_and_is_deterministic():
    v = [1.0, 2.0, 3.0, 4.0, 10.0]
    m, ci = ms.boot_mean_ci(v, n_boot=2000)
    assert m == pytest.approx(4.0) and ci[0] <= m <= ci[1]
    assert ms.boot_mean_ci(v, n_boot=2000) == (m, ci)


def test_pooled_rate_counts_passes_over_events():
    rate, ci, k, n = ms.pooled_rate_ci([[[1, True], [3, False]], [[2, True]]], n_boot=500)
    assert (k, n) == (2, 3) and rate == pytest.approx(2 / 3) and ci[0] <= ci[1]


def test_summarize_paired_differences_and_counts():
    runs = [
        fake_run(0, 3, 6, [[1, True], [3, False]]),
        fake_run(1, 4, 5, [[3, False]]),
        fake_run(2, 3, 6, [[2, True], [3, False]]),
    ]
    s = ms.summarize(runs)
    c = s["configs"]["default"]
    assert s["n"] == 3 and c["breaker_open"] == 3
    assert c["accepted_diff"][0] == pytest.approx((3 + 1 + 3) / 3)
    assert c["accepted_diff_min_max"] == [1, 3]
    assert c["pass_rate"][2:] == (2, 5)
    assert s["stock"]["breaker_open"] == 3
    assert set(c["policies"]) == set(ms.POLICIES)


def test_run_scores_match_hand_computation():
    run = {"events": [[1, True]]}
    tf, pa, br = ms.run_scores(run, {"kind": "fixed", "p": 0.75})
    assert tf == pytest.approx(75.0 - 0.8)  # 0.5*75 + 0.5*75 - 0.8
    assert pa == pytest.approx(75.0)
    assert br == pytest.approx(0.0625)


def test_splice_is_idempotent_and_replaces_only_its_block():
    doc = "# Title\n\ntext\n"
    once = ms.splice(doc, "## A\nfirst")
    twice = ms.splice(once, "## A\nsecond")
    assert "first" not in twice and "second" in twice and twice.startswith("# Title")
    assert twice.count(ms.BEGIN) == 1
    assert ms.splice(twice, "## A\nsecond") == twice


def test_collect_seed_matches_a_direct_run_and_render_runs():
    r = ms.collect_seed(42)
    assert r["stock"]["steps"] == 8
    inj = r["injected"]["default"]
    assert (inj["steps"], inj["accepted"]) == (11, 6)
    assert sum(1 for _, p in inj["events"] if p) == 3
    text = ms.render(sorted([r, ms.collect_seed(0)], key=lambda x: x["seed"]))
    assert "seeds 0 to 42" in text and "Honest limits" in text
