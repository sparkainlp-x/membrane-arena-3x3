"""Reproducible benchmark suite for Membrane Arena 3x3 (SYNTHETIC).

Produces parameter-response curves, stability maps and falsifiable
hypotheses — no physical, quantum-hardware, biological, medical or
consciousness claim is made. Outputs are deterministic: the same
arguments always produce byte-identical CSV/JSON (no timestamps, no
wall-clock fields, no unseeded randomness). Plots are optional and
require matplotlib; CSV/JSON are always written.

Usage:
    python scripts/benchmark.py --out-dir benchmark_results --seeds 0-99
    python scripts/benchmark.py --out-dir benchmark_results --seeds 0-4 \
        --steps 200 --dt-values 0.01,0.05,0.1 --coupling-values 0.0,0.02,0.10 \
        --operator-modes legacy,exact --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.lattice import LatticeConfig
from membrane_arena_3x3.shields import GuardConfig


def parse_seeds(text: str) -> list[int]:
    """Parse '0-99' or '0,1,42' into a sorted seed list."""
    if "-" in text and "," not in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return sorted({int(x) for x in text.split(",") if x.strip()})


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def run_one(seed: int, steps: int, **overrides) -> dict:
    """Run one arena step by step, recording the breaker-open step."""
    lattice_kwargs = {}
    for key in ("coupling", "dephasing", "mirror_strength", "dt", "operator_mode"):
        if key in overrides:
            lattice_kwargs[key] = overrides[key]
    guard_kwargs = {}
    for key in (
        "residual_target",
        "divergence_min",
        "divergence_max",
        "purity_floor",
        "entropy_target",
    ):
        if key in overrides:
            guard_kwargs[key] = overrides[key]
    config = EngineConfig(
        lattice=LatticeConfig(**lattice_kwargs),
        guards=GuardConfig(**guard_kwargs),
        steps=steps,
        seed=seed,
        schumann=overrides.get("schumann", False),
        failure_limit=overrides.get("failure_limit", 5),
        adaptive_retry=overrides.get("adaptive_retry", True),
        max_retries=overrides.get("max_retries", 3),
    )
    arena = MembraneArena3x3(config)
    breaker_step = None
    for _ in range(steps):
        result = arena.step()
        if result["status"] == "circuit_breaker_open":
            breaker_step = arena.current_step
            break
    summary = arena.summary()
    final = summary["final"] or {}
    accepted = summary["accepted_steps"]
    executed = summary["steps_executed"]
    return {
        "seed": seed,
        "steps": steps,
        "steps_executed": executed,
        "accepted_steps": accepted,
        "rejected_steps": summary["rejected_steps"],
        "accepted_fraction": accepted / executed if executed else 0.0,
        "status": summary["status"],
        "breaker_step": breaker_step,
        "survived": summary["status"] == "completed",
        "repair_accepts": summary["repair_accepts"],
        "repair_attempts_total": summary["repair_attempts_total"],
        "final_max_residual": final.get("max_residual"),
        "final_min_purity": final.get("min_purity"),
        "final_max_divergence": final.get("max_divergence"),
        "final_mean_divergence": final.get("mean_divergence"),
        "final_mean_purity": final.get("mean_purity"),
        "final_mean_off_diagonal_coherence": final.get(
            "mean_off_diagonal_coherence"
        ),
        "final_stability_penalty": final.get("stability_penalty"),
        "guards": summary["guards"],
        "operator_mode": summary["operator_mode"],
        **{k: overrides[k] for k in sorted(overrides) if k not in ("schumann",)},
    }


def survival_curve(rows: list[dict], steps: int) -> list[float]:
    """Fraction of runs still open (not breaker-latched) after each step."""
    n = len(rows)
    curve = []
    for step in range(steps + 1):
        alive = sum(
            1
            for r in rows
            if r["breaker_step"] is None or r["breaker_step"] > step
        )
        curve.append(alive / n if n else 0.0)
    return curve


def summarize(rows: list[dict]) -> dict:
    """Mean / min / max of the numeric columns, plus survival fractions."""
    out: dict = {"n_runs": len(rows)}
    for key in (
        "steps_executed",
        "accepted_steps",
        "accepted_fraction",
        "final_max_residual",
        "final_min_purity",
        "final_max_divergence",
        "final_stability_penalty",
    ):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_min"] = float(np.min(vals))
            out[f"{key}_max"] = float(np.max(vals))
    out["survived_fraction"] = (
        sum(1 for r in rows if r["survived"]) / len(rows) if rows else 0.0
    )
    out["breaker_fraction"] = 1.0 - out["survived_fraction"]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    flat = []
    for r in rows:
        row = {k: v for k, v in r.items() if k != "guards"}
        row["guards"] = json.dumps(r["guards"], sort_keys=True)
        flat.append(row)
    if not flat:
        return
    keys = list(flat[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)


def make_plots(out_dir: Path, curve_by_label: dict[str, list[float]], steps: int) -> list[str]:
    """Write survival-curve plots; return [] if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    fig, ax = plt.subplots()
    xs = list(range(steps + 1))
    for label, curve in curve_by_label.items():
        ax.plot(xs, curve, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction of runs still open")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.set_title("Membrane Arena 3x3 — survival curves (SYNTHETIC)")
    path = out_dir / "survival_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [str(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="benchmark_results")
    parser.add_argument("--seeds", default="0-99")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt-values", default="0.05")
    parser.add_argument("--coupling-values", default="0.02")
    parser.add_argument("--operator-modes", default="legacy")
    parser.add_argument("--adaptive-retry", dest="adaptive_retry", action="store_true", default=True)
    parser.add_argument("--no-adaptive-retry", dest="adaptive_retry", action="store_false")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    curves: dict[str, list[float]] = {}
    sweep: dict[str, dict] = {}
    for dt in parse_floats(args.dt_values):
        for coupling in parse_floats(args.coupling_values):
            for mode in args.operator_modes.split(","):
                mode = mode.strip()
                label = f"dt={dt} coupling={coupling} mode={mode}"
                rows = [
                    run_one(
                        seed,
                        args.steps,
                        dt=dt,
                        coupling=coupling,
                        operator_mode=mode,
                        adaptive_retry=args.adaptive_retry,
                    )
                    for seed in seeds
                ]
                all_rows.extend(rows)
                curves[label] = survival_curve(rows, args.steps)
                sweep[label] = summarize(rows)

    write_csv(out_dir / "runs.csv", all_rows)
    plots = [] if args.no_plots else make_plots(out_dir, curves, args.steps)
    result = {
        "seeds": seeds,
        "steps": args.steps,
        "adaptive_retry": args.adaptive_retry,
        "sweep": sweep,
        "survival_curves": curves,
        "plots": plots,
        "n_runs": len(all_rows),
    }
    with open(out_dir / "benchmark.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"n_runs": len(all_rows), "plots": plots, "sweep": sweep}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
