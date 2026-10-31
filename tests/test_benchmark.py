"""Benchmark determinism and hypothesis checks (SYNTHETIC, small sweeps)."""

import json

from scripts.benchmark import parse_seeds, run_one, survival_curve, summarize


def test_seed_parsing():
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("42,0,42") == [0, 42]


def test_benchmark_run_is_deterministic():
    a = run_one(42, 20)
    b = run_one(42, 20)
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_adaptive_retry_benefit_is_measurable_in_benchmark():
    with_retry = run_one(42, 200, adaptive_retry=True)
    without = run_one(42, 200, adaptive_retry=False)
    assert with_retry["accepted_steps"] > without["accepted_steps"]
    assert with_retry["repair_accepts"] > 0


def test_survival_curve_is_monotone_and_bounded():
    rows = [run_one(seed, 50) for seed in range(5)]
    curve = survival_curve(rows, 50)
    assert len(curve) == 51
    assert curve[0] == 1.0
    assert all(0.0 <= v <= 1.0 for v in curve)
    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1))


def test_summarize_reports_survival_fraction():
    rows = [run_one(seed, 20) for seed in range(3)]
    summary = summarize(rows)
    assert summary["n_runs"] == 3
    assert 0.0 <= summary["survived_fraction"] <= 1.0
    assert summary["breaker_fraction"] == 1.0 - summary["survived_fraction"]
