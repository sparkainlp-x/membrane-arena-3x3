"""Measure the stock arena against the opt-in Crystallize injection (SYNTHETIC).

Usage:  python scripts/injection_sweep.py [--write docs/injection_results.md]

Runs seeds 42, 0, 1, 2, 3 with 200 steps requested and prints a Markdown
table. A second table is an ablation: the same variants with the mirror left
at its stock strength, so only the smaller dt differs. Every number is
computed here on the machine that runs it; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import platform
import sys

import numpy as np

from membrane_arena_3x3.crystallize_injection import (
    InjectionConfig,
    run_simulation_with_injection,
)
from membrane_arena_3x3.engine import EngineConfig, run_simulation

SEEDS = (42, 0, 1, 2, 3)
STEPS = 200


def rows(injection: InjectionConfig | None, statuses: list[str]) -> list[str]:
    out = []
    for seed in SEEDS:
        base = run_simulation(EngineConfig(steps=STEPS, seed=seed))
        inj = run_simulation_with_injection(
            EngineConfig(steps=STEPS, seed=seed), injection
        )
        statuses.append(inj["status"])
        out.append(
            f"| {seed} | {base['steps_executed']} / {base['accepted_steps']} "
            f"| {inj['steps_executed']} / {inj['accepted_steps']} "
            f"| {inj['injection_passes']} of {inj['injection_events']} "
            f"| {inj['injection_score_total']:.1f} "
            f"| {inj['injection_cost_total']:.1f} | {inj['status']} |"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", default=None, help="also write the Markdown here")
    args = parser.parse_args()

    statuses: list[str] = []
    header = (
        "| seed | base steps / accepted | injected steps / accepted "
        "| injection passes of events | injection score (points) "
        "| sampling cost | final status |\n|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Injection sweep (SYNTHETIC, measured by scripts/injection_sweep.py)",
        "",
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"{STEPS} steps requested, default `InjectionConfig` "
        "(3 attempts, commit p = 0.75).",
        "",
        "## Default injection (dt shrink + mirror softened to 0.0)",
        "",
        header,
        *rows(None, statuses),
        "",
        "## Ablation: dt shrink only (mirror left at the stock 0.10)",
        "",
        header,
        *rows(InjectionConfig(mirror_variant=0.10), statuses),
        "",
        "The injection never changes a guard, threshold or the breaker. "
        f"Injected runs ending with the breaker open: {statuses.count('circuit_breaker_open')} "
        f"of {len(statuses)}; the injection does not cure the arena. 'Injection score' is the sum of Crystallize points "
        "(100 * (1 - 4 * Brier)) over every variant tried; it is negative "
        "because most variants fail the guards while the commit is 0.75.",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.write:
        with open(args.write, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    sys.exit(main())
