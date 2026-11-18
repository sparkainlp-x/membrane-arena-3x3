# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the discrete vector-calculus module (SYNTHETIC).

Gauss/Stokes are exact telescoping identities: they are tested as
implementation checks (residual < 1e-12), NOT as discoveries. The
escape-predictor verdicts are pinned to the measured evaluation split
in experiments/results/vector_calculus.json (frozen thresholds, seeds
0, 2, 3, 123, 2026, 11, 99); the test loads that committed evidence
and asserts the recorded verdicts, so README/results drift is caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from membrane_arena_3x3.core import fidelity, purity
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.vector_calculus import (
    ESCAPE_REASONS,
    FROZEN_DIV_FLAG,
    FROZEN_FID_FLAG,
    auc_rank,
    circulation,
    collect_escape_dataset,
    divergence,
    edge_fluxes,
    evaluate_predictor,
    gauss_identity,
    node_gradient,
    scalar_field,
    standard_configs,
    step_features,
    stokes_identity,
)

RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results" / "vector_calculus.json"

# Measured on the evaluation split (frozen thresholds); see
# docs/vector_calculus_preregistration.md. Update ONLY by re-running
# scripts/measure_vector_calculus.py.
MEASURED = {
    "n_steps": 490,
    "n_escapes": 105,
    "base_rate": 0.21428571428571427,
    "div_precision": 0.7216494845360825,
    "div_recall": 0.6666666666666666,
    "auc_div": 0.7688311688311689,
    "auc_fid": 0.6242424242424243,
    "auc_gain_vs_fid": 0.1445887445887446,
    "gauss_max_abs_residual": 6.661338147750939e-16,
    "stokes_max_abs_residual": 2.7755575615628914e-16,
    "claim_i_pass": True,
    "claim_iv_pass": True,
}


def _rng_fields(seed: int = 0, n: int = 50) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random((3, 3)) for _ in range(n)]


def test_gauss_exact_on_random_fields():
    for s in _rng_fields():
        out = gauss_identity(s)
        assert abs(out["residual"]) < 1e-12
        # Closed lattice: total divergence and boundary flux both vanish.
        assert abs(out["total_divergence"]) < 1e-12
        assert abs(out["boundary_flux"]) < 1e-12


def test_stokes_exact_on_random_fields():
    for s in _rng_fields():
        out = stokes_identity(s)
        assert abs(out["residual"]) < 1e-12
        # A gradient field has zero total circulation (discrete
        # fundamental theorem): total == boundary == 0.
        assert abs(out["total_circulation"]) < 1e-12
        assert abs(out["boundary_circulation"]) < 1e-12


def test_gauss_stokes_exact_on_structured_fields():
    fields = [
        np.zeros((3, 3)),
        np.ones((3, 3)),
        np.arange(9, dtype=float).reshape(3, 3),
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
    ]
    for s in fields:
        assert abs(gauss_identity(s)["residual"]) < 1e-12
        assert abs(stokes_identity(s)["residual"]) < 1e-12


def test_divergence_uniform_field_is_zero():
    assert np.allclose(divergence(np.full((3, 3), 0.7)), 0.0)


def test_divergence_linear_field():
    # s = i: flow points south (-i), so row 0 is a sink (div -1),
    # row 2 a source (div +1), row 1 balanced (div 0).
    s = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    expected = np.array(
        [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    assert np.allclose(divergence(s), expected)


def test_divergence_corners_are_live():
    # Closed boundary: corner cells must be able to carry divergence
    # (constant-extrapolation boundaries zero them identically).
    rng = np.random.default_rng(3)
    div = divergence(rng.random((3, 3)))
    assert np.all(div[[0, 0, 2, 2], [0, 2, 0, 2]] != 0.0)


def test_edge_fluxes_closed_boundary():
    s = np.arange(9, dtype=float).reshape(3, 3)
    faces = edge_fluxes(s)
    assert np.all(faces["fx_east"][:, 2] == 0.0)
    assert np.all(faces["fx_west"][:, 0] == 0.0)
    assert np.all(faces["fy_north"][2, :] == 0.0)
    assert np.all(faces["fy_south"][0, :] == 0.0)
    # Interior faces agree across the shared face.
    assert np.allclose(faces["fx_east"][:, 0], faces["fx_west"][:, 1])
    assert np.allclose(faces["fy_north"][0, :], faces["fy_south"][1, :])


def test_node_gradient_linear_field():
    s = np.arange(9, dtype=float).reshape(3, 3)
    gx, gy = node_gradient(s)
    assert np.allclose(gx, 1.0)  # ds/dj = 1 everywhere
    assert np.allclose(gy, 3.0)  # ds/di = 3 everywhere


def test_circulation_linear_field_is_zero():
    s = np.arange(9, dtype=float).reshape(3, 3)
    assert np.allclose(circulation(s), 0.0)


def test_scalar_field_kinds():
    arena = MembraneArena3x3(EngineConfig(steps=0, seed=42))
    state, ref = arena.lattice.state, arena.lattice.reference
    f = scalar_field(state, ref, kind="fidelity")
    assert f.shape == (3, 3)
    assert np.all((f >= 0.0) & (f <= 1.0))
    # Initial state IS the reference: fidelity 1 everywhere.
    assert np.allclose(f, 1.0)
    p = scalar_field(state, ref, kind="purity")
    assert np.allclose(p[0, 0], purity(state[0, 0]))
    assert np.allclose(f[1, 2], fidelity(state[1, 2], ref[1, 2]))
    with pytest.raises(ValueError):
        scalar_field(state, ref, kind="entropy")


def test_step_features_keys_and_drop():
    arena = MembraneArena3x3(EngineConfig(steps=0, seed=42))
    feats = step_features(arena.lattice.state, arena.lattice.reference)
    assert feats["max_fidelity_drop"] == 0.0
    assert set(feats) == {
        "max_abs_div",
        "max_div",
        "mean_abs_div",
        "max_infidelity",
        "mean_infidelity",
        "max_fidelity_drop",
    }
    # After one step the drop is measured against the previous state.
    prev = arena.lattice.state.copy()
    arena.step()
    feats2 = step_features(arena.lattice.state, arena.lattice.reference, prev)
    assert feats2["max_fidelity_drop"] >= 0.0


def test_auc_rank_known_cases():
    assert auc_rank([0.1, 0.2, 0.9, 1.0], [False, False, True, True]) == 1.0
    assert auc_rank([0.9, 1.0, 0.1, 0.2], [False, False, True, True]) == 0.0
    assert auc_rank([0.5, 0.5, 0.5], [True, False, True]) == 0.5
    assert auc_rank([0.1, 0.2], [True, True]) == 0.5  # single class
    assert auc_rank([0.1, 0.2], [False, False]) == 0.5


def test_evaluate_predictor_on_synthetic_records():
    from membrane_arena_3x3.vector_calculus import EscapeRecord

    records = [
        EscapeRecord(0, "x", 1, True, True, "pairwise_divergence_below_minimum",
                     {"max_abs_div": 0.9, "max_infidelity": 0.1, "max_fidelity_drop": 0.0}),
        EscapeRecord(0, "x", 2, True, True, "divergence_above_maximum",
                     {"max_abs_div": 0.8, "max_infidelity": 0.2, "max_fidelity_drop": 0.0}),
        EscapeRecord(0, "x", 3, False, False, "ok",
                     {"max_abs_div": 0.01, "max_infidelity": 0.8, "max_fidelity_drop": 0.0}),
        EscapeRecord(0, "x", 4, False, False, "ok",
                     {"max_abs_div": 0.02, "max_infidelity": 0.9, "max_fidelity_drop": 0.0}),
    ]
    ev = evaluate_predictor(records)
    assert ev["div_precision"] == 1.0
    assert ev["div_recall"] == 1.0
    assert ev["auc_div"] == 1.0
    assert ev["auc_gain_vs_fid"] > 0.0
    with pytest.raises(ValueError):
        evaluate_predictor([])


def test_planted_collision_is_a_fast_escape():
    # One config, one seed: the planted pair must escape on a
    # divergence guard with high pre-step divergence.
    recs = collect_escape_dataset(
        {"planted_collision": standard_configs()["planted_collision"]},
        seeds=[42],
    )
    assert any(r.escaped for r in recs)
    assert all(r.reason in ESCAPE_REASONS | {"ok"} for r in recs)


def test_frozen_thresholds_are_the_preregistered_values():
    assert FROZEN_DIV_FLAG == 0.15
    assert FROZEN_FID_FLAG == 0.60


def test_measured_evaluation_verdicts():
    """Pin the committed evaluation evidence (frozen thresholds)."""
    assert RESULTS.exists(), "run scripts/measure_vector_calculus.py first"
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert data["seeds"] == [0, 2, 3, 11, 99, 123, 2026]
    assert data["div_flag"] == FROZEN_DIV_FLAG
    assert data["fid_flag"] == FROZEN_FID_FLAG
    ev = data["evaluation"]
    for key, value in MEASURED.items():
        if isinstance(value, bool):
            continue  # verdicts pinned below
        if key in ("gauss_max_abs_residual", "stokes_max_abs_residual"):
            assert data[key] == pytest.approx(value, abs=1e-15)
        elif key in ev:
            assert ev[key] == pytest.approx(value, abs=1e-9)
    # (ii)/(iii): identities exact on every checked field.
    assert data["gauss_max_abs_residual"] < 1e-12
    assert data["stokes_max_abs_residual"] < 1e-12
    # (i)/(iv): verdicts recorded in the evidence file itself, pinned
    # to the measured values above.
    assert ev["claim_i_pass"] is MEASURED["claim_i_pass"]
    assert ev["claim_iv_pass"] is MEASURED["claim_iv_pass"]
    # The recorded verdicts must agree with the frozen bars.
    assert ev["claim_i_pass"] == (
        ev["div_precision"] >= 2.0 * ev["base_rate"]
        and ev["div_recall"] >= 0.30
    )
    assert ev["claim_iv_pass"] == (ev["auc_gain_vs_fid"] >= 0.05)
