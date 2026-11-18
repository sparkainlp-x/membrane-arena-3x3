# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
"""Run the frozen Lyapunov protocol (see REPORT.md) and write JSON results.

Claims:
  (i)  sign flip at 0.90: targets {0.80, 0.85, 0.88, 0.92, 0.95, 0.98}
       x seeds {42, 1, 7}, blend + matched Hamiltonians, 25 steps.
  (ii) seed-1 excursion: trio 0.85-start, row, m=2.5, seeds 1 vs 42/7,
       plus the seed-1 column control and m=1.0 context runs.
  (iii) half-dt: matched H, F0=0.85, seeds {42, 1, 7},
       dt=0.05 (25 steps) vs dt=0.025 (50 steps), same 1.25 time units.

Usage: python3 scripts/lyapunov_sweep.py [--out-dir lyapunov_results]
Deterministic: fixed seeds, no timestamps in the JSON payload.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from membrane_arena_3x3.lattice import LatticeConfig
from membrane_arena_3x3.lyapunov import (
    blend_to_mean_fidelity,
    fit_lambda,
    setup_matched_hamiltonians,
    setup_trio,
    twin_run,
)

ROW0 = [(0, 0), (0, 1), (0, 2)]
COL0 = [(0, 0), (1, 0), (2, 0)]
TARGETS = (0.80, 0.85, 0.88, 0.92, 0.95, 0.98)
SEEDS = (42, 1, 7)
WINDOW_STEPS = 25


def _mean_fit(run: dict, n_steps: int) -> dict:
    fit = fit_lambda(
        np.asarray(run["times"]),
        np.asarray(run["mean_trace_distances"]),
        window_steps=n_steps,
    )
    return {
        "lambda_mean": fit.lam,
        "lambda_mean_r_squared": fit.r_squared,
        "lambda_mean_n_points": fit.n_points,
        "lambda_mean_fitted": fit.fitted,
    }


def _crossing_fit(run: dict) -> dict:
    """Sensitivity: fit truncated at the first step mean fidelity
    crosses 0.90 from its initial side (None if it never crosses)."""
    mf = run["mean_pairwise_fidelity"]
    f0 = run["initial_mean_fidelity"]
    cross = next(
        (
            k
            for k in range(1, len(mf))
            if (f0 > 0.90 and mf[k] <= 0.90)
            or (f0 < 0.90 and mf[k] >= 0.90)
        ),
        None,
    )
    if cross is None or cross < 5:
        return {"crossing_step": cross, "lambda_cross": None}
    fit = fit_lambda(
        np.asarray(run["times"]),
        np.asarray(run["trace_distances"]),
        window_steps=cross,
    )
    return {
        "crossing_step": cross,
        "lambda_cross": fit.lam,
        "lambda_cross_r_squared": fit.r_squared,
        "lambda_cross_n_points": fit.n_points,
    }


def _record(run, n_steps=WINDOW_STEPS, **extra):
    d = run.to_dict()
    d.update(_mean_fit(d, n_steps))
    d.update(extra)
    return d


def _claim_i():
    rows = []
    for seed in SEEDS:
        for target in TARGETS:
            def setup(lat, target=target):
                blend_to_mean_fidelity(lat, target)
                setup_matched_hamiltonians(lat)

            run = twin_run(
                LatticeConfig(), seed=seed, steps=WINDOW_STEPS,
                setup=setup, fit_window_steps=WINDOW_STEPS,
            )
            rec = _record(run, claim="i", target_fidelity=target)
            rec.update(_crossing_fit(rec))
            rows.append(rec)
            print(
                f"(i) seed={seed} F0={target:.2f} achieved={run.initial_mean_fidelity:.4f} "
                f"lam_max={run.fit.lam:+.4f} (R2={run.fit.r_squared:.2f}) "
                f"lam_mean={rec['lambda_mean']:+.4f} "
                f"cross={rec['crossing_step']} lam_cross={rec['lambda_cross']} "
                f"d25={run.trace_distances[-1]:.2e}",
                flush=True,
            )
    return rows


def _trio_config(cells, mult):
    mult9 = [1.0] * 9
    for (i, j) in cells[1:]:
        mult9[3 * i + j] = mult
    return LatticeConfig(operator_mode="legacy", dt_multipliers=tuple(mult9))


def _claim_ii():
    rows = []
    for label, seed, cells, mult in [
        ("seed1-row-m2.5", 1, ROW0, 2.5),
        ("seed42-row-m2.5", 42, ROW0, 2.5),
        ("seed7-row-m2.5", 7, ROW0, 2.5),
        ("seed1-col-m2.5", 1, COL0, 2.5),
        ("seed1-row-m1.0-context", 1, ROW0, 1.0),
        ("seed42-row-m1.0-context", 42, ROW0, 1.0),
    ]:
        def setup(lat, cells=cells):
            setup_trio(lat, cells, 0.85)

        run = twin_run(
            _trio_config(cells, mult), seed=seed, steps=WINDOW_STEPS,
            setup=setup, fit_window_steps=WINDOW_STEPS,
        )
        rec = _record(run, claim="ii", label=label)
        rows.append(rec)
        print(
            f"(ii) {label}: lam_max={run.fit.lam:+.4f} (R2={run.fit.r_squared:.2f}) "
            f"lam_mean={rec['lambda_mean']:+.4f} d25={run.trace_distances[-1]:.2e}",
            flush=True,
        )
    return rows


def _claim_iii():
    rows = []
    for seed in SEEDS:
        for dt, n_steps in [(0.05, 25), (0.025, 50)]:
            def setup(lat):
                blend_to_mean_fidelity(lat, 0.85)
                setup_matched_hamiltonians(lat)

            run = twin_run(
                LatticeConfig(dt=dt), seed=seed, steps=n_steps,
                setup=setup, fit_window_steps=n_steps,
            )
            rec = _record(run, n_steps=n_steps, claim="iii", dt=dt, seed=seed)
            # Sensitivity: same-step-count fit (25 steps) at dt/2.
            if dt == 0.025:
                f25 = fit_lambda(
                    np.asarray(run.times),
                    np.asarray(run.trace_distances),
                    window_steps=25,
                )
                rec["lambda_25steps"] = f25.lam
                rec["lambda_25steps_r_squared"] = f25.r_squared
            rows.append(rec)
            print(
                f"(iii) seed={seed} dt={dt}: lam_max={run.fit.lam:+.4f} "
                f"(R2={run.fit.r_squared:.2f}) lam_mean={rec['lambda_mean']:+.4f} "
                f"horizon={run.fit.horizon} d_end={run.trace_distances[-1]:.2e}",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="lyapunov_results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "protocol": {
            "delta0": 1e-6,
            "perturb_seed": 20260717,
            "window_steps": WINDOW_STEPS,
            "operator_mode": "legacy",
            "coupling": 0.02,
            "dephasing": 0.5,
            "mirror_strength": 0.10,
            "seeds": list(SEEDS),
            "targets_i": list(TARGETS),
            "margin_ii": 0.10,
            "tolerance_iii": 0.15,
        },
        "claim_i": _claim_i(),
        "claim_ii": _claim_ii(),
        "claim_iii": _claim_iii(),
    }
    path = out_dir / "lyapunov_final.json"
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
