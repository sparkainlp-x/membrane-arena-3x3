"""Collect raw per-run data for the injection analysis (SYNTHETIC).

For each regime and seed this runs the stock engine and the injected engine
once. The commit policy changes only the ledger, so one injected run per seed
serves every policy; policies are scored offline in ``experiments.rescoring``.
Raw runs are cached as JSON lines under ``experiments/results/``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

from membrane_arena_3x3.crystallize_injection import (
    InjectionConfig,
    MembraneArena3x3WithInjection,
)
from membrane_arena_3x3.engine import MembraneArena3x3

from experiments.regimes import REGIMES

DEV_SEEDS = tuple(range(1000, 1100))
CONFIRM_SEEDS = tuple(range(2000, 2100))
TUNING_SEEDS = (42, 0, 1, 2, 3)
SETS = {"dev": DEV_SEEDS, "confirm": CONFIRM_SEEDS, "tuning": TUNING_SEEDS}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def collect_run(regime_name: str, seed: int) -> dict:
    """One seed in one regime: stock run plus injected run, reduced to what scoring needs."""
    reg = REGIMES[regime_name]
    stock = MembraneArena3x3(reg.engine_config(seed))
    stock_summary = stock.run()
    eng = MembraneArena3x3WithInjection(
        reg.engine_config(seed),
        InjectionConfig(first_factor=reg.first_factor, mirror_variant=reg.mirror_variant),
    )
    s = eng.run()
    return {
        "regime": regime_name,
        "seed": seed,
        "stock_steps": stock_summary["steps_executed"],
        "stock_accepted": stock_summary["accepted_steps"],
        "stock_status": stock_summary["status"],
        "steps": s["steps_executed"],
        "accepted": s["accepted_steps"],
        "status": s["status"],
        "engine_score": s["injection_score_total"],
        "events": [[len(e["attempts"]), bool(e["passed"])] for e in eng.injections],
        "base_reasons": [e["base_reason"] for e in eng.injections],
    }


def _job(args):
    return collect_run(*args)


def collect_set(set_name: str, workers: int | None = None, use_cache: bool = True) -> list[dict]:
    """All regimes for one seed set, cached in experiments/results/runs_<set>.jsonl."""
    path = os.path.join(RESULTS_DIR, f"runs_{set_name}.jsonl")
    if use_cache and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]
    jobs = [(r, s) for r in REGIMES for s in SETS[set_name]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(_job, jobs, chunksize=1))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r) + "\n")
    return runs
