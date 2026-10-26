"""Pre-registered analysis of the Crystallize injection (SYNTHETIC).

Implements docs/PREREGISTRATION-injection.md. Usage:
    python -m experiments.run_prereg_analysis [--write]
Writes experiments/results/prereg_results.json and prereg_tables.md with --write.
Every number in docs/RESULTS-injection.md that comes from this analysis is in
those files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np

from experiments import rescoring as rs
from experiments import stats
from experiments.injection_runs import (
    CONFIRM_SEEDS,
    DEV_SEEDS,
    RESULTS_DIR,
    TUNING_SEEDS,
    collect_set,
)
from experiments.regimes import REGIMES

REGIME_NAMES = list(REGIMES)


def runs_by_regime(runs):
    return {r: sorted((x for x in runs if x["regime"] == r), key=lambda x: x["seed"]) for r in REGIME_NAMES}


def pooled_pass_rate(runs):
    ev = sum(len(x["events"]) for x in runs)
    ps = sum(sum(1 for _, p in x["events"] if p) for x in runs)
    return ps / ev if ev else float("nan"), ps, ev


def dev_tuned_p(dev_runs) -> float:
    rate, _, _ = pooled_pass_rate(dev_runs)
    return float(min(0.99, max(0.01, round(rate, 2))))


def policies(ft: float) -> dict:
    return {
        "F35": {"kind": "fixed", "p": 0.35},
        "F75": {"kind": "fixed", "p": 0.75},
        "A35": {"kind": "adaptive", "first": 0.35, "weight": 4},
        "A75": {"kind": "adaptive", "first": 0.75, "weight": 4},
        "FT": {"kind": "fixed", "p": ft},
    }


def run_brier(run, policy) -> float | None:
    """Mean first-forecast Brier of one run; None if the run has no events."""
    if not run["events"]:
        return None
    firsts = rs.policy_first_commits(run["events"], policy)
    return float(np.mean(rs.first_forecast_brier(run["events"], firsts)))


def compare(diffs):
    return {**stats.paired_effect(diffs), "p": stats.sign_flip_p(diffs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--set", default="confirm", choices=("confirm", "dev"), help="dev is for code checks only")
    args = ap.parse_args()

    dev = runs_by_regime(collect_set("dev"))
    conf = runs_by_regime(collect_set(args.set))
    tune = runs_by_regime(collect_set("tuning"))
    out = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "seeds": {"dev": [DEV_SEEDS[0], DEV_SEEDS[-1]], "confirm": [CONFIRM_SEEDS[0], CONFIRM_SEEDS[-1]], "tuning": list(TUNING_SEEDS)},
        "regimes": {},
    }
    confirmatory = []  # (id, regime, label, result)

    for ri, reg in enumerate(REGIME_NAMES):
        pols = policies(dev_tuned_p(dev[reg]))
        d_rate = pooled_pass_rate(dev[reg])
        c_rate = pooled_pass_rate(conf[reg])
        block = {
            "dev_pass_rate": d_rate[0], "dev_events": d_rate[2],
            "confirm_pass_rate": c_rate[0], "confirm_events": c_rate[2],
            "FT_p": pols["FT"]["p"],
        }
        cr = conf[reg]
        n = len(cr)
        block["breaker_open"] = {
            "k": sum(1 for x in cr if x["status"] == "circuit_breaker_open"), "n": n,
        }
        block["breaker_open"]["wilson95"] = stats.wilson(block["breaker_open"]["k"], n)
        block["stock_breaker_open_k"] = sum(1 for x in cr if x["stock_status"] == "circuit_breaker_open")
        reasons: dict[str, int] = {}
        for x in cr:
            for r_ in x["base_reasons"]:
                reasons[r_] = reasons.get(r_, 0) + 1
        block["first_rejection_reasons"] = reasons
        block["mean_steps"] = float(np.mean([x["steps"] for x in cr]))
        block["mean_accepted"] = float(np.mean([x["accepted"] for x in cr]))
        block["mean_stock_steps"] = float(np.mean([x["stock_steps"] for x in cr]))
        block["mean_stock_accepted"] = float(np.mean([x["stock_accepted"] for x in cr]))
        block["runs_without_events"] = sum(1 for x in cr if not x["events"])

        # C1-C3: accepted steps, injection minus none
        acc = [x["accepted"] - x["stock_accepted"] for x in cr]
        confirmatory.append((f"C{1 + ri}", reg, "accepted steps: injection minus no injection", compare(acc)))
        block["steps_diff"] = stats.paired_effect([x["steps"] - x["stock_steps"] for x in cr])

        # Brier tables and C4-C12
        with_ev = [x for x in cr if x["events"]]
        br = {k: [run_brier(x, p) for x in with_ev] for k, p in pols.items()}
        block["mean_first_forecast_brier"] = {k: float(np.mean(v)) for k, v in br.items()}
        diff = lambda a, b: np.array(br[a]) - np.array(br[b])
        confirmatory.append((f"C{4 + ri}", reg, "Brier F35 minus F75", compare(diff("F35", "F75"))))
        confirmatory.append((f"C{7 + ri}", reg, "Brier A35 minus F35", compare(diff("A35", "F35"))))
        confirmatory.append((f"C{10 + ri}", reg, "Brier A35 minus FT", compare(diff("A35", "FT"))))
        block["exploratory_Brier_A75_minus_F75"] = stats.paired_effect(diff("A75", "F75"))

        # calibration, pooled
        block["calibration"] = {}
        for k, p in pols.items():
            fs, ys = [], []
            for x in with_ev:
                fs.append(rs.policy_first_commits(x["events"], p))
                ys.append([int(o) for _, o in x["events"]])
            block["calibration"][k] = stats.ece_with_ci(fs, ys)

        # ledger scores, both scorings (exploratory)
        led = {}
        for k, p in pols.items():
            a = [rs.ledger_per_attempt(x["events"], rs.policy_first_commits(x["events"], p)) for x in cr]
            b = [rs.ledger_two_forecast(x["events"], rs.policy_first_commits(x["events"], p)) for x in cr]
            led[k] = {"per_attempt_mean": float(np.mean(a)), "per_attempt_ci": stats.paired_effect(a)["mean_ci"],
                      "two_forecast_mean": float(np.mean(b)), "two_forecast_ci": stats.paired_effect(b)["mean_ci"]}
        block["ledger_per_run"] = led

        # tuning seeds, in-sample
        tr = tune[reg]
        block["tuning_seeds_total_ledger"] = {
            k: {"per_attempt": float(sum(rs.ledger_per_attempt(x["events"], rs.policy_first_commits(x["events"], p)) for x in tr)),
                "two_forecast": float(sum(rs.ledger_two_forecast(x["events"], rs.policy_first_commits(x["events"], p)) for x in tr))}
            for k, p in pols.items()
        }
        block["tuning_seeds_steps_accepted"] = [[x["seed"], x["steps"], x["accepted"], x["stock_steps"], x["stock_accepted"]] for x in tr]
        out["regimes"][reg] = block

    holm = stats.holm([c[3]["p"] for c in confirmatory])
    out["confirmatory"] = [
        {"id": c[0], "regime": c[1], "comparison": c[2], **c[3], "p_holm": h["p_holm"], "reject_h0_at_0.05": h["reject"]}
        for c, h in zip(confirmatory, holm)
    ]
    text = render(out)
    print(text)
    if args.write and args.set != "confirm":
        raise SystemExit("--write is only allowed for the confirm set")
    if args.write:
        with open(os.path.join(RESULTS_DIR, "prereg_results.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        with open(os.path.join(RESULTS_DIR, "prereg_tables.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(out) -> str:
    L = ["# Pre-registered analysis output (SYNTHETIC, generated by experiments/run_prereg_analysis.py)", "",
         f"Python {out['environment']['python']}, NumPy {out['environment']['numpy']}. Analysis set is the confirm set for the committed results. Seeds: dev {out["seeds"]["dev"]}, confirm {out['seeds']['confirm']}, tuning {out['seeds']['tuning']}.", ""]
    L += ["## Per-regime descriptives (analysis set, 100 runs per regime)", "",
          "| regime | dev pass rate (events) | confirm pass rate (events) | FT commit | breaker open, injected (Wilson 95%) | breaker open, stock | mean steps stock | mean steps injected | mean accepted stock | mean accepted injected | runs without events |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, b in out["regimes"].items():
        bo = b["breaker_open"]
        L.append(f"| {r} | {f(b['dev_pass_rate'])} ({b['dev_events']}) | {f(b['confirm_pass_rate'])} ({b['confirm_events']}) | {b['FT_p']:.2f} | {bo['k']}/{bo['n']} ({f(bo['wilson95'][0])} to {f(bo['wilson95'][1])}) | {b['stock_breaker_open_k']}/{bo['n']} | {f(b['mean_stock_steps'],1)} | {f(b['mean_steps'],1)} | {f(b['mean_stock_accepted'],1)} | {f(b['mean_accepted'],1)} | {b['runs_without_events']} |")
    L += ["", "First rejection reasons (analysis set, count of events): " + "; ".join(f"{r}: {b['first_rejection_reasons']}" for r, b in out["regimes"].items()), ""]
    L += ["## Confirmatory tests (family of 12, Holm), analysis set", "",
          "| id | regime | comparison | mean difference [95% CI] | dz [95% CI] | p (sign-flip) | p Holm | reject at 0.05 |", "|---|---|---|---|---|---|---|---|"]
    for c in out["confirmatory"]:
        dz = "n/a" if c["dz"] is None else f"{c['dz']:.2f} [{f(c['dz_ci'][0],2)}, {f(c['dz_ci'][1],2)}]"
        L.append(f"| {c['id']} | {c['regime']} | {c['comparison']} | {c['mean']:.4f} [{c['mean_ci'][0]:.4f}, {c['mean_ci'][1]:.4f}] | {dz} | {c['p']:.5f} | {c['p_holm']:.5f} | {'yes' if c['reject_h0_at_0.05'] else 'no'} |")
    L += ["", "## Mean first-forecast Brier per run (analysis set; lower is better)", "",
          "| regime | F35 | F75 | A35 | A75 | FT |", "|---|---|---|---|---|---|"]
    for r, b in out["regimes"].items():
        m = b["mean_first_forecast_brier"]
        L.append(f"| {r} | {m['F35']:.4f} | {m['F75']:.4f} | {m['A35']:.4f} | {m['A75']:.4f} | {m['FT']:.4f} |")
    L += ["", "## Pooled expected calibration error, 5 bins (analysis set; cluster bootstrap 95% CI)", "",
          "| regime | F35 | F75 | A35 | A75 | FT |", "|---|---|---|---|---|---|"]
    for r, b in out["regimes"].items():
        c = b["calibration"]
        L.append(f"| {r} | " + " | ".join(f"{c[k]['ece']:.3f} [{c[k]['ece_ci'][0]:.3f}, {c[k]['ece_ci'][1]:.3f}]" for k in ("F35", "F75", "A35", "A75", "FT")) + " |")
    L += ["", "### Reliability tables, adaptive from 0.35 (A35)", ""]
    for r, b in out["regimes"].items():
        L += [f"{r}:", "", "| bin | events | mean forecast | observed pass rate |", "|---|---|---|---|"]
        for t in b["calibration"]["A35"]["reliability"]:
            L.append(f"| {t['bin']} | {t['n']} | {f(t['mean_forecast'])} | {f(t['observed_rate'])} |")
        L.append("")
    L += ["## Ledger score per run, mean [95% bootstrap CI] (exploratory, analysis set)", "",
          "Per-attempt scoring is the merged engine's; two-forecast scoring is the design note's (cost of 0.8 per variant subtracted).", "",
          "| regime | scoring | F35 | F75 | A35 | A75 | FT |", "|---|---|---|---|---|---|---|"]
    for r, b in out["regimes"].items():
        for sc, key in (("per-attempt", "per_attempt"), ("two-forecast", "two_forecast")):
            L.append(f"| {r} | {sc} | " + " | ".join(
                f"{b['ledger_per_run'][k][key + '_mean']:.1f} [{b['ledger_per_run'][k][key + '_ci'][0]:.1f}, {b['ledger_per_run'][k][key + '_ci'][1]:.1f}]"
                for k in ("F35", "F75", "A35", "A75", "FT")) + " |")
    L += ["", "## Tuning seeds 42, 0, 1, 2, 3 (in-sample; total ledger over the 5 runs)", "",
          "| regime | scoring | F35 | F75 | A35 | A75 | FT |", "|---|---|---|---|---|---|---|"]
    for r, b in out["regimes"].items():
        for sc in ("per_attempt", "two_forecast"):
            L.append(f"| {r} | {sc} | " + " | ".join(f"{b['tuning_seeds_total_ledger'][k][sc]:.1f}" for k in ("F35", "F75", "A35", "A75", "FT")) + " |")
    L += ["", "## Steps difference, injection minus none (analysis set, executed steps; mean [95% CI])", ""]
    for r, b in out["regimes"].items():
        e = b["steps_diff"]
        L.append(f"- {r}: {e['mean']:.2f} [{e['mean_ci'][0]:.2f}, {e['mean_ci'][1]:.2f}]")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
