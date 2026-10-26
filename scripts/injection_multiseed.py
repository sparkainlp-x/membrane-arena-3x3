"""Multi-seed measurement of the Crystallize injection (SYNTHETIC, stock regime).

Usage (from the repository root):
    OMP_NUM_THREADS=1 python scripts/injection_multiseed.py [--seeds 50] [--write]

For seeds 0 .. N-1 (200 steps requested) it runs the stock engine once and the
injected engine under two settings:

* ``default``  : dt shrink plus the mirror softened to 0.0 (the module default);
* ``dt_only``  : dt shrink only, the mirror left at the stock 0.10 (ablation).

The commit policy changes only the ledger, never acceptance, so one injected
run per seed and setting serves every commit policy; the policies (fixed 0.35,
fixed 0.75, adaptive from 0.35, adaptive from 0.75) are scored offline with
``experiments.rescoring``, which is pinned to the engine's own ledger by tests.
Both scorings are reported: the engine's two-forecast scoring (author's
original) and the per-attempt scoring (every variant scored) that the first
merged version used. Intervals are 95% percentile bootstrap intervals over
seeds (10,000 resamples). Every number is computed here; nothing is hard-coded.
With ``--write`` the section between the multiseed markers of
docs/injection_results.md is replaced (or appended), and the raw per-seed runs
are written to experiments/results/multiseed_runs.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import rescoring as rs  # noqa: E402
from experiments import stats  # noqa: E402
from membrane_arena_3x3.crystallize_injection import (  # noqa: E402
    InjectionConfig,
    MembraneArena3x3WithInjection,
)
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3  # noqa: E402

STEPS = 200
CONFIGS = {
    "default": dict(mirror_variant=0.0),
    "dt_only": dict(mirror_variant=0.10),
}
POLICIES = {
    "fixed 0.35": {"kind": "fixed", "p": 0.35},
    "fixed 0.75": {"kind": "fixed", "p": 0.75},
    "adaptive from 0.35": {"kind": "adaptive", "first": 0.35, "weight": 4},
    "adaptive from 0.75": {"kind": "adaptive", "first": 0.75, "weight": 4},
}
BEGIN = "<!-- multiseed:start -->"
END = "<!-- multiseed:end -->"


def collect_seed(seed: int) -> dict:
    """Stock run plus one injected run per setting for one seed."""
    stock = MembraneArena3x3(EngineConfig(steps=STEPS, seed=seed)).run()
    out = {
        "seed": seed,
        "stock": {
            "steps": stock["steps_executed"],
            "accepted": stock["accepted_steps"],
            "status": stock["status"],
        },
        "injected": {},
    }
    for name, kw in CONFIGS.items():
        eng = MembraneArena3x3WithInjection(
            EngineConfig(steps=STEPS, seed=seed), InjectionConfig(**kw)
        )
        s = eng.run()
        out["injected"][name] = {
            "steps": s["steps_executed"],
            "accepted": s["accepted_steps"],
            "status": s["status"],
            "engine_score_total": s["injection_score_total"],
            "events": [[len(e["attempts"]), bool(e["passed"])] for e in eng.injections],
        }
    return out


def collect(seeds, workers: int | None = None) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(collect_seed, seeds, chunksize=1))


# ---------------------------------------------------------------- statistics
def boot_mean_ci(values, n_boot: int = stats.N_BOOT, seed: int = stats.BOOT_SEED):
    """Mean and 95% percentile bootstrap interval of a per-seed sample."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def pooled_rate_ci(runs_events, n_boot: int = stats.N_BOOT, seed: int = stats.BOOT_SEED):
    """Pooled pass rate over events with a cluster (by seed) bootstrap interval."""
    k = np.array([sum(1 for _, p in ev if p) for ev in runs_events], dtype=float)
    n = np.array([len(ev) for ev in runs_events], dtype=float)
    rate = k.sum() / n.sum() if n.sum() else float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), size=(n_boot, len(k)))
    den = n[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(den > 0, k[idx].sum(axis=1) / den, np.nan)
    return float(rate), [float(np.nanpercentile(rates, 2.5)), float(np.nanpercentile(rates, 97.5))], int(k.sum()), int(n.sum())


def fmt(m, ci, nd=1):
    return f"{m:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def run_scores(run, policy):
    ev = run["events"]
    firsts = rs.policy_first_commits(ev, policy)
    brier = float(np.mean(rs.first_forecast_brier(ev, firsts))) if ev else None
    return rs.ledger_two_forecast(ev, firsts), rs.ledger_per_attempt(ev, firsts), brier


def summarize(runs: list[dict]) -> dict:
    """All reported statistics, from the raw per-seed runs."""
    n = len(runs)
    out = {"n": n, "stock": {}, "configs": {}}
    out["stock"]["breaker_open"] = sum(r["stock"]["status"] == "circuit_breaker_open" for r in runs)
    out["stock"]["steps"] = boot_mean_ci([r["stock"]["steps"] for r in runs])
    out["stock"]["accepted"] = boot_mean_ci([r["stock"]["accepted"] for r in runs])
    for name in CONFIGS:
        inj = [r["injected"][name] for r in runs]
        c = {}
        c["breaker_open"] = sum(x["status"] == "circuit_breaker_open" for x in inj)
        c["breaker_wilson"] = stats.wilson(c["breaker_open"], n)
        c["steps"] = boot_mean_ci([x["steps"] for x in inj])
        c["accepted"] = boot_mean_ci([x["accepted"] for x in inj])
        c["steps_diff"] = boot_mean_ci([x["steps"] - r["stock"]["steps"] for x, r in zip(inj, runs)])
        c["accepted_diff"] = boot_mean_ci([x["accepted"] - r["stock"]["accepted"] for x, r in zip(inj, runs)])
        c["accepted_diff_min_max"] = [
            min(x["accepted"] - r["stock"]["accepted"] for x, r in zip(inj, runs)),
            max(x["accepted"] - r["stock"]["accepted"] for x, r in zip(inj, runs)),
        ]
        c["events_per_run"] = boot_mean_ci([len(x["events"]) for x in inj])
        c["pass_rate"] = pooled_rate_ci([x["events"] for x in inj])
        c["runs_with_no_events"] = sum(1 for x in inj if not x["events"])
        c["policies"] = {}
        for pname, pol in POLICIES.items():
            with_ev = [x for x in inj if x["events"]]
            sc = [run_scores(x, pol) for x in inj]
            c["policies"][pname] = {
                "two_forecast": boot_mean_ci([s[0] for s in sc]),
                "per_attempt": boot_mean_ci([s[1] for s in sc]),
                "brier": boot_mean_ci([s[2] for s in sc if s[2] is not None]) if with_ev else None,
                "two_forecast_negative_runs": sum(1 for s in sc if s[0] < 0),
            }
        out["configs"][name] = c
    return out


# ------------------------------------------------------------------ rendering
def render(runs: list[dict]) -> str:
    s = summarize(runs)
    n = s["n"]
    d35 = s["configs"]["default"]["policies"]["fixed 0.35"]
    seeds = f"{runs[0]['seed']} to {runs[-1]['seed']}"
    L = [
        f"## Multi-seed sweep, seeds {seeds} ({n} seeds, stock regime)",
        "",
        f"Measured by `scripts/injection_multiseed.py` on Python {platform.python_version()}, NumPy {np.__version__}, "
        f"{STEPS} steps requested per run, 3 injection attempts, injection RNG `default_rng(seed + 10000007)`. "
        "Intervals are 95% percentile bootstrap intervals over seeds (10,000 resamples). "
        "One injected run per seed and setting serves every commit policy, because the commit changes only the ledger.",
        "",
        "Stock engine (no injection): "
        f"breaker open in {s['stock']['breaker_open']}/{n} runs, "
        f"steps {fmt(*s['stock']['steps'])}, accepted steps {fmt(*s['stock']['accepted'])} (mean [95% CI]).",
        "",
        "### Steps, acceptance and breaker, injected versus stock (paired by seed)",
        "",
        "| setting | steps injected | accepted injected | accepted, injected minus stock | steps, injected minus stock | breaker open (Wilson 95%) | events per run | pooled pass rate [cluster 95% CI] (passes / events) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    labels = {"default": "default (dt shrink + mirror 0.0)", "dt_only": "ablation: dt shrink only (mirror 0.10)"}
    for name, c in s["configs"].items():
        pr, prci, k, ev = c["pass_rate"]
        L.append(
            f"| {labels[name]} | {fmt(*c['steps'])} | {fmt(*c['accepted'])} | {fmt(*c['accepted_diff'], nd=2)} "
            f"(range {c['accepted_diff_min_max'][0]} to {c['accepted_diff_min_max'][1]}) | {fmt(*c['steps_diff'], nd=2)} "
            f"| {c['breaker_open']}/{n} ({c['breaker_wilson'][0]:.3f} to {c['breaker_wilson'][1]:.3f}) "
            f"| {fmt(*c['events_per_run'])} | {pr:.3f} [{prci[0]:.3f}, {prci[1]:.3f}] ({k} / {ev}) |"
        )
    L += [
        "",
        "### Ledger score per run by commit policy, mean [95% CI]",
        "",
        "Two-forecast scoring is the engine's (0.5 points of the first commit + 0.5 points of the last commit "
        "- 0.8 per variant tried). Per-attempt scoring (every variant scored, cost not subtracted) is the scoring "
        "the first merged version used; it is computed offline from the same events and shown for comparison. "
        "The last column is the mean first-forecast Brier score (lower is better) over runs that had events.",
        "",
    ]
    for name, c in s["configs"].items():
        L += [
            f"{labels[name]}:",
            "",
            "| commit policy | two-forecast score | per-attempt score | mean first-forecast Brier | runs with two-forecast score < 0 |",
            "|---|---|---|---|---|",
        ]
        for pname, p in c["policies"].items():
            b = "n/a" if p["brier"] is None else fmt(*p["brier"], nd=3)
            L.append(
                f"| {pname} | {fmt(*p['two_forecast'])} | {fmt(*p['per_attempt'])} | {b} "
                f"| {p['two_forecast_negative_runs']}/{n} |"
            )
        L.append("")
    L += [
        "### Honest limits",
        "",
        "- This is one toy model (a synthetic 3x3 lattice of 32-dimensional density matrices) at its published default "
        "parameters, on one machine and one NumPy. Nothing here says anything about a physical system.",
        f"- The {n} seeds are a convenience set (0 to {n - 1}) that includes the five seeds on which the 0.35 default "
        "was chosen. The 0.35 rows are therefore partly in-sample. The pre-registered analysis uses disjoint seeds.",
        "- The sweep shows what trying shrunk variants after a rejection changes in this model: the accepted-step "
        "and steps columns above, and a ledger that measures calibration. It does not show that the injection cures "
        "the arena. Breaker open at the end of the run, injected: "
        + ", ".join(f"{labels[k]} {c['breaker_open']}/{n}" for k, c in s["configs"].items())
        + f"; stock {s['stock']['breaker_open']}/{n}.",
        "- Forecasts do not use event features, so the ledger measures calibration to a base rate, not event-level "
        "skill. A score depends on how close the commit is to the pass rate and on the scoring rule chosen. For the "
        "default setting and a fixed 0.35 commit the mean score is "
        f"{d35['per_attempt'][0]:.1f} under per-attempt scoring and {d35['two_forecast'][0]:.1f} "
        f"[{d35['two_forecast'][1][0]:.1f}, {d35['two_forecast'][1][1]:.1f}] under the engine's two-forecast scoring "
        "(per-attempt scoring counts every failed variant as a separate miss and subtracts no cost; the two-forecast "
        "rule scores two forecasts against the event outcome and subtracts 0.8 per variant), so a score is not an "
        "outcome measure by itself and the sign of a policy's score depends on the scoring rule.",
        "- Within-run events are not independent (the last events of every run are failures by construction, because "
        "the breaker needs consecutive rejections), so the pooled pass rate is a descriptive rate, and the intervals "
        "above resample seeds, not events.",
        "- The intervals are uncorrected for multiplicity and are for magnitude, not for a decision.",
        "",
    ]
    return "\n".join(L)


def splice(doc: str, section: str) -> str:
    block = f"{BEGIN}\n{section.rstrip()}\n{END}\n"
    if BEGIN in doc and END in doc:
        a, b = doc.index(BEGIN), doc.index(END) + len(END)
        tail = doc[b:].lstrip("\n")
        return doc[:a] + block + (("\n" + tail) if tail else "")
    return doc.rstrip("\n") + "\n\n" + block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50, help="seeds 0 .. N-1")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="re-render from experiments/results/multiseed_runs.jsonl")
    args = ap.parse_args(argv)

    runs_path = os.path.join(ROOT, "experiments", "results", "multiseed_runs.jsonl")
    if args.render_only:
        with open(runs_path, encoding="utf-8") as fh:
            runs = [json.loads(line) for line in fh]
    else:
        runs = collect(range(args.seeds), args.workers)
    runs = sorted(runs, key=lambda r: r["seed"])
    text = render(runs)
    print(text)
    if args.write:
        res_dir = os.path.join(ROOT, "experiments", "results")
        os.makedirs(res_dir, exist_ok=True)
        if not args.render_only:
            with open(runs_path, "w", encoding="utf-8") as fh:
                for r in runs:
                    fh.write(json.dumps(r) + "\n")
        doc_path = os.path.join(ROOT, "docs", "injection_results.md")
        with open(doc_path, encoding="utf-8") as fh:
            doc = fh.read()
        with open(doc_path, "w", encoding="utf-8") as fh:
            fh.write(splice(doc, text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
