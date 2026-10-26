"""Offline scoring of commit policies from recorded events (SYNTHETIC).

An event is ``(n_attempts, passed)``. Attempts 1..n-1 failed, attempt n passed
or not; the commit before attempt k is the first commit revised k-1 times
halfway toward 0.5. Two scorings are implemented:

* ``ledger_two_forecast``: the engine's scoring (author's original), two
  forecasts per event minus 0.8 per variant. ``tests/test_experiments.py`` pins
  it to the engine's own ledger.
* ``ledger_per_attempt``: every variant scored, cost not subtracted. This is
  what the first merged version of the engine did; it is kept so both scorings
  can be reported side by side, and a test pins it to the engine's attempt records.
"""

from __future__ import annotations

from membrane_arena_3x3.crystallize_injection import (
    brier,
    calibration_points,
    clamp_p,
    revise_toward_humility,
)

COST_PER_VARIANT = 0.8


def policy_first_commits(events, policy: dict) -> list[float]:
    """First commit of each event under a policy.

    policy = {"kind": "fixed", "p": x} or
             {"kind": "adaptive", "first": x, "weight": w}.
    """
    if policy["kind"] == "fixed":
        return [clamp_p(policy["p"])] * len(events)
    first, w = policy["first"], policy["weight"]
    out, passes = [], 0
    for i, (_, passed) in enumerate(events):
        denom = i + w
        out.append(clamp_p(first if denom == 0 else (passes + w * first) / denom))
        passes += int(passed)
    return out


def attempt_commits(p1: float, n: int) -> list[float]:
    ps, p = [], p1
    for _ in range(n):
        ps.append(p)
        p = clamp_p(revise_toward_humility(p))
    return ps


def ledger_per_attempt(events, firsts) -> float:
    """Per-attempt scoring (first merged version): sum of points over every variant tried, cost not subtracted."""
    total = 0.0
    for (n, passed), p1 in zip(events, firsts):
        for k, p in enumerate(attempt_commits(p1, n)):
            total += calibration_points(p, passed and k == n - 1)
    return total


def ledger_two_forecast(events, firsts) -> float:
    """Engine scoring (author's original): mean of first and last commit points vs the event outcome, minus 0.8 per variant."""
    total = 0.0
    for (n, passed), p1 in zip(events, firsts):
        ps = attempt_commits(p1, n)
        total += (
            0.5 * calibration_points(ps[0], passed)
            + 0.5 * calibration_points(ps[-1], passed)
            - COST_PER_VARIANT * n
        )
    return total


def first_forecast_brier(events, firsts) -> list[float]:
    return [brier(p, passed) for (_, passed), p in zip(events, firsts)]
