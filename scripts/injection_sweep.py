"""Measure the stock arena against the opt-in Crystallize injection (SYNTHETIC).

Usage:  python scripts/injection_sweep.py [--write docs/injection_results.md]

Runs seeds 42, 0, 1, 2, 3 with 200 steps requested and prints a Markdown
table. Further tables compare the commit policies (fixed 0.75, fixed 0.35,
adaptive from 0.35, adaptive from 0.75; they change only the ledger), an
ablation with the mirror left at its stock strength, and an evolved genome
supplied with the design note (a different regime). Every number is computed
here on the machine that runs it; nothing is hard-coded.
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
from membrane_arena_3x3.lattice import LatticeConfig

SEEDS = (42, 0, 1, 2, 3)
STEPS = 200
EVOLVED_SEEDS = (42, 0, 1)
EVOLVED = LatticeConfig(
    coupling=0.0339, dephasing=0.2871, mirror_strength=0.1725, dt=0.0128
)
POLICIES = (
    ("fixed 0.75", dict(first_commit=0.75)),
    ("fixed 0.35", dict(first_commit=0.35)),
    ("adaptive from 0.35", dict(first_commit=0.35, adaptive=True)),
    ("adaptive from 0.75", dict(first_commit=0.75, adaptive=True)),
)


def rows(injection: InjectionConfig | None, statuses: list[str]) -> list[str]:
    out = []
    for seed in SEEDS:
        base = run_simulation(EngineConfig(steps=STEPS, seed=seed))
        inj = run_simulation_with_injection(
            EngineConfig(steps=STEPS, seed=seed), injection
        )
        statuses.append(inj["status"])
        s075 = run_simulation_with_injection(
            EngineConfig(steps=STEPS, seed=seed),
            InjectionConfig(
                **{**(vars(injection) if injection else {}), "first_commit": 0.75}
            ),
        )["injection_score_total"]
        out.append(
            f"| {seed} | {base['steps_executed']} / {base['accepted_steps']} "
            f"| {inj['steps_executed']} / {inj['accepted_steps']} "
            f"| {inj['injection_passes']} of {inj['injection_events']} "
            f"| {inj['injection_score_total']:.1f} | {s075:.1f} "
            f"| {inj['injection_cost_total']:.1f} | {inj['status']} |"
        )
    return out


def policy_table(seeds, lattice: LatticeConfig | None, extra: dict) -> list[str]:
    """Ledger score per commit policy; the dynamics are identical across policies."""
    names = [name for name, _ in POLICIES]
    lines = [
        "| seed | steps / accepted | passes of events | "
        + " | ".join(names)
        + " | adaptive final commit (from 0.35) |",
        "|---|---|---|" + "---|" * (len(names) + 1),
    ]
    totals = [0.0] * len(names)
    for seed in seeds:
        scores, dyn, final = [], None, None
        for k, (name, kw) in enumerate(POLICIES):
            cfg = (
                EngineConfig(steps=STEPS, seed=seed, lattice=lattice)
                if lattice
                else EngineConfig(steps=STEPS, seed=seed)
            )
            res = run_simulation_with_injection(cfg, InjectionConfig(**kw, **extra))
            scores.append(res["injection_score_total"])
            totals[k] += res["injection_score_total"]
            key = (res["steps_executed"], res["accepted_steps"], res["injection_passes"], res["injection_events"])
            assert dyn in (None, key), "commit policy changed the dynamics"
            dyn = key
            if name == "adaptive from 0.35":
                final = res["adaptive_pass_rate"]
        lines.append(
            f"| {seed} | {dyn[0]} / {dyn[1]} | {dyn[2]} of {dyn[3]} | "
            + " | ".join(f"{x:.1f}" for x in scores)
            + f" | {final:.3f} |"
        )
    lines.append(
        "| total | | | " + " | ".join(f"{x:.1f}" for x in totals) + " | |"
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", default=None, help="also write the Markdown here")
    args = parser.parse_args()

    statuses: list[str] = []
    header = (
        "| seed | base steps / accepted | injected steps / accepted "
        "| injection passes of events | event-score total (commit 0.35) "
        "| event-score total (commit 0.75) "
        "| sampling cost | final status |\n|---|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Injection sweep (SYNTHETIC, measured by scripts/injection_sweep.py)",
        "",
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"{STEPS} steps requested, default `InjectionConfig` "
        "(3 attempts, fixed commit 0.35). Injection RNG: default_rng(seed + 10000007), "
        "one uniform(0.8, 1.2) draw per attempt.",
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
        "## Commit policy, stock regime (score in points; dynamics identical across policies)",
        "",
        *policy_table(SEEDS, None, {}),
        "",
        "## Commit policy, evolved genome supplied with the design note (a different regime)",
        "",
        "Genome: coupling 0.0339, dephasing 0.2871, mirror_strength 0.1725, dt 0.0128, "
        "first_factor 0.2555, mirror_variant 0.0. Seeds 42, 0, 1.",
        "",
        *policy_table(EVOLVED_SEEDS, EVOLVED, dict(first_factor=0.2555, mirror_variant=0.0)),
        "",
        "The injection never changes a guard, threshold or the breaker. "
        f"Injected runs in the first two tables ending with the breaker open: {statuses.count('circuit_breaker_open')} "
        f"of {len(statuses)}; the injection does not cure the arena. 'Injection score' is the sum of Crystallize points "
        "(100 * (1 - 4 * Brier)): per event 0.5 * points(first commit) + 0.5 * points(last commit) "
        "minus 0.8 per variant tried (two forecasts per event, not every variant). "
        "The commit changes only this ledger. "
        "A fixed commit scores well only where it is near the regime's pass rate; "
        "the adaptive commit tracks the rate but pays a burn-in cost.",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.write:
        # Keep the multi-seed section written by scripts/injection_multiseed.py, if present.
        try:
            with open(args.write, encoding="utf-8") as fh:
                old = fh.read()
        except FileNotFoundError:
            old = ""
        begin, end = "<!-- multiseed:start -->", "<!-- multiseed:end -->"
        if begin in old and end in old:
            block = old[old.index(begin) : old.index(end) + len(end)]
            text = text.rstrip("\n") + "\n\n" + block + "\n"
        with open(args.write, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    sys.exit(main())
