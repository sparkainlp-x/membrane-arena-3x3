# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
"""Measure the vector-calculus claims on the EVALUATION split (SYNTHETIC).

Usage:
    python scripts/measure_vector_calculus.py
    python scripts/measure_vector_calculus.py --seeds 0,2,3,123,2026 --out experiments/results/vector_calculus.json

Thresholds are FROZEN (see docs/vector_calculus_preregistration.md and
the FROZEN_* constants in membrane_arena_3x3/vector_calculus.py): they
were set on the calibration split (seeds 42, 1, 7) and are never moved
here. Output is deterministic: same arguments, byte-identical JSON
(no timestamps, no wall-clock fields, no unseeded randomness).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from membrane_arena_3x3.vector_calculus import (
    FROZEN_DIV_FLAG,
    FROZEN_FID_FLAG,
    collect_escape_dataset,
    divergence,
    evaluate_predictor,
    gauss_identity,
    scalar_field,
    standard_configs,
    stokes_identity,
)

# Evaluation split: seeds NOT used for calibration (calibration: 42, 1, 7).
EVAL_SEEDS = [0, 2, 3, 123, 2026, 11, 99]


def parse_seeds(text: str) -> list[int]:
    return sorted({int(x) for x in text.split(",") if x.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in EVAL_SEEDS),
        help="comma-separated evaluation seeds (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="experiments/results/vector_calculus.json",
        help="output JSON path (default: %(default)s)",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    configs = standard_configs()

    # (ii)/(iii) identity checks on random fields AND on real lattice fields.
    rng = np.random.default_rng(1234)
    gauss_residuals: list[float] = []
    stokes_residuals: list[float] = []
    for _ in range(200):
        s = rng.random((3, 3))
        gauss_residuals.append(abs(gauss_identity(s)["residual"]))
        stokes_residuals.append(abs(stokes_identity(s)["residual"]))
    # Real lattice fields: one stock run, every accepted state.
    from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3

    arena = MembraneArena3x3(EngineConfig(steps=200, seed=42))
    prev = None
    real_div_max: list[float] = []
    while not arena.circuit_breaker and arena.current_step < 200:
        s = scalar_field(arena.lattice.state, arena.lattice.reference)
        gauss_residuals.append(abs(gauss_identity(s)["residual"]))
        stokes_residuals.append(abs(stokes_identity(s)["residual"]))
        real_div_max.append(float(np.max(np.abs(divergence(s)))))
        prev = arena.lattice.state.copy()
        arena.step()
    del prev

    records = collect_escape_dataset(configs, seeds=seeds)
    evaluation = evaluate_predictor(records)

    result = {
        "version": "0.2.1",
        "seeds": seeds,
        "configs": sorted(configs),
        "div_flag": FROZEN_DIV_FLAG,
        "fid_flag": FROZEN_FID_FLAG,
        "gauss_max_abs_residual": float(max(gauss_residuals)),
        "stokes_max_abs_residual": float(max(stokes_residuals)),
        "n_identity_fields": len(gauss_residuals),
        "real_lattice_max_abs_div": float(max(real_div_max)) if real_div_max else 0.0,
        "real_lattice_steps": len(real_div_max),
        "evaluation": evaluation,
        "records": [r.to_dict() for r in records],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ev = evaluation
    print(f"steps={ev['n_steps']} escapes={ev['n_escapes']} base_rate={ev['base_rate']:.4f}")
    print(
        f"(i) div precision={ev['div_precision']:.3f} recall={ev['div_recall']:.3f} "
        f"lift={ev['precision_lift']:.2f}x claim_i_pass={ev['claim_i_pass']}"
    )
    print(
        f"(ii) gauss max|resid|={result['gauss_max_abs_residual']:.2e} "
        f"(iii) stokes max|resid|={result['stokes_max_abs_residual']:.2e}"
    )
    print(
        f"(iv) auc_div={ev['auc_div']:.3f} auc_fid={ev['auc_fid']:.3f} "
        f"auc_drop={ev['auc_drop']:.3f} gain_vs_fid={ev['auc_gain_vs_fid']:.3f} "
        f"claim_iv_pass={ev['claim_iv_pass']}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
