# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
"""Finite-time Lyapunov exponent: twin-run measurement.

Pins the mechanism in membrane_arena_3x3/lyapunov.py: trace-distance
twins, the trace-preserving perturbation, the log-linear fit, and the
twin-run determinism. The measured claims themselves live in REPORT.md
(they are measurement outcomes, not unit-test assertions); the tests
here pin only the machinery and one smoke measurement.
"""

import inspect

import numpy as np
import pytest

import membrane_arena_3x3.lyapunov as lyap
from membrane_arena_3x3.core import (
    fidelity,
    is_valid_density_matrix,
    project_valid,
    random_density_matrix,
)
from membrane_arena_3x3.lattice import LatticeConfig
from membrane_arena_3x3.lyapunov import (
    blend_to_mean_fidelity,
    fit_lambda,
    mean_pairwise_fidelity,
    mean_twin_trace_distance,
    perturb_toward,
    setup_trio,
    trace_distance,
    twin_run,
    twin_trace_distance,
)


def _pure_state(rng, dim=4):
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return np.outer(vec, vec.conj())


def test_trace_distance_basics():
    rng = np.random.default_rng(0)
    rho = random_density_matrix(rng, 4)
    assert trace_distance(rho, rho) == pytest.approx(0.0, abs=1e-12)
    sigma = random_density_matrix(rng, 4)
    d = trace_distance(rho, sigma)
    assert 0.0 < d <= 1.0
    assert trace_distance(sigma, rho) == pytest.approx(d)
    # Orthogonal pure states are at the maximum, 1.0.
    e0 = np.zeros((4, 4), dtype=complex)
    e0[0, 0] = 1.0
    e1 = np.zeros((4, 4), dtype=complex)
    e1[1, 1] = 1.0
    assert trace_distance(e0, e1) == pytest.approx(1.0, abs=1e-12)
    # Maximally mixed vs pure: 1 - 1/dim.
    mixed = np.eye(4, dtype=complex) / 4
    assert trace_distance(e0, mixed) == pytest.approx(0.75, abs=1e-12)


def test_perturb_toward_exact_delta0_and_valid():
    rng = np.random.default_rng(1)
    rho = random_density_matrix(rng, 32)
    sigma = random_density_matrix(rng, 32)
    perturbed = perturb_toward(rho, sigma, 1e-6)
    assert trace_distance(rho, perturbed) == pytest.approx(1e-6, rel=1e-6)
    assert is_valid_density_matrix(perturbed)
    assert abs(np.trace(perturbed).real - 1.0) < 1e-12
    # A larger delta0 is honoured exactly too.
    perturbed2 = perturb_toward(rho, sigma, 1e-3)
    assert trace_distance(rho, perturbed2) == pytest.approx(1e-3, rel=1e-6)


def test_perturb_toward_rejects_too_close_target():
    rng = np.random.default_rng(2)
    rho = random_density_matrix(rng, 32)
    sigma = project_valid(rho + 1e-9 * random_density_matrix(rng, 32))
    assert trace_distance(rho, sigma) < 1e-6
    with pytest.raises(ValueError):
        perturb_toward(rho, sigma, 1e-6)


def test_fit_lambda_recovers_known_slope():
    times = np.arange(0, 26) * 0.05
    dists = 1e-6 * np.exp(2.0 * times)
    fit = fit_lambda(times, dists, window_steps=25)
    assert fit.fitted
    assert fit.lam == pytest.approx(2.0, rel=1e-6)
    assert fit.n_points == 25
    assert fit.horizon == pytest.approx(np.log(1e6) / 2.0, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fit_lambda_contracting_has_no_horizon():
    times = np.arange(0, 26) * 0.05
    dists = 1e-6 * np.exp(-0.5 * times)
    fit = fit_lambda(times, dists, window_steps=25)
    assert fit.lam == pytest.approx(-0.5, rel=1e-6)
    assert fit.horizon is None


def test_fit_lambda_below_noise_floor_is_not_fitted():
    times = np.arange(0, 26) * 0.05
    # 1 - fidelity for a 1e-6 perturbation sits at ~1e-12: no fit.
    dists = np.full(26, 4e-12)
    fit = fit_lambda(times, dists)
    assert not fit.fitted
    assert fit.lam is None


def test_fit_lambda_window_steps_truncates_to_available_points():
    times = np.arange(0, 11) * 0.05
    dists = 1e-6 * np.exp(0.3 * times)
    fit = fit_lambda(times, dists, window_steps=25)
    assert fit.n_points == 10
    assert fit.lam == pytest.approx(0.3, rel=1e-6)


def test_blend_to_mean_fidelity_hits_target():
    from membrane_arena_3x3.lattice import MembraneLattice

    lat = MembraneLattice(LatticeConfig(), np.random.default_rng(42))
    assert mean_pairwise_fidelity(lat.state) < 0.70
    achieved = blend_to_mean_fidelity(lat, 0.85)
    assert achieved == pytest.approx(0.85, abs=0.005)
    assert mean_pairwise_fidelity(lat.state) == pytest.approx(0.85, abs=0.005)
    for i in range(3):
        for j in range(3):
            assert is_valid_density_matrix(lat.state[i, j])


def test_setup_trio_pins_fidelity_and_matches_hamiltonians():
    from membrane_arena_3x3.lattice import MembraneLattice

    cells = [(0, 0), (0, 1), (0, 2)]
    lat = MembraneLattice(LatticeConfig(), np.random.default_rng(1))
    setup_trio(lat, cells, 0.85)
    for other in cells[1:]:
        assert fidelity(lat.state[cells[0]], lat.state[other]) == pytest.approx(
            0.85, abs=0.01
        )
    for i, j in cells:
        assert np.allclose(
            lat.hamiltonians[i, j], lat.hamiltonians[cells[0]]
        )


def test_twin_run_deterministic_and_starts_at_delta0():
    cfg = LatticeConfig()
    run_a = twin_run(cfg, seed=42, steps=5)
    run_b = twin_run(cfg, seed=42, steps=5)
    assert run_a.trace_distances == run_b.trace_distances
    assert run_a.times == run_b.times
    assert run_a.trace_distances[0] == pytest.approx(1e-6, rel=1e-6)
    assert run_a.steps_executed == 5
    assert len(run_a.mean_pairwise_fidelity) == 6
    assert np.isfinite(run_a.fit.lam)
    # Mean never exceeds max, step by step.
    for mean_d, max_d in zip(
        run_a.mean_trace_distances, run_a.trace_distances
    ):
        assert mean_d <= max_d + 1e-15


def test_twin_run_single_cell_perturbation_option():
    cfg = LatticeConfig()
    run = twin_run(cfg, seed=42, steps=5, perturb_cells=((0, 0),))
    assert run.trace_distances[0] == pytest.approx(1e-6, rel=1e-6)
    assert np.isfinite(run.fit.lam)


def test_twin_run_rejects_bad_args():
    with pytest.raises(ValueError):
        twin_run(LatticeConfig(), seed=42, steps=0)
    with pytest.raises(ValueError):
        twin_run(LatticeConfig(), seed=42, steps=5, delta0=0.0)
    with pytest.raises(ValueError):
        twin_run(LatticeConfig(), seed=42, steps=5, delta0=1.5)


def test_module_does_not_reuse_lyapunov_proxy():
    # No import of shields (where lyapunov_proxy / stability_penalty
    # live) anywhere in the module. The docstring names lyapunov_proxy
    # only to state that it is NOT reused, so the word itself is
    # allowed in prose; what is pinned here is the absence of any
    # import or attribute use.
    source = inspect.getsource(lyap)
    assert "from .shields import" not in source
    assert "from membrane_arena_3x3.shields import" not in source
    assert "import shields" not in source
    assert not hasattr(lyap, "lyapunov_proxy")
    assert not hasattr(lyap, "stability_penalty")


def test_twin_run_smoke_stock_seed42():
    # Smoke measurement (machinery check, not a claim): the stock
    # seed-42 map over 25 steps has a finite fitted exponent -- measured
    # +0.86 (R^2 0.94), positive, as the random lattice sits well below
    # the 0.90 threshold -- and the infidelity check series stays at its
    # numerical floor, so it is reported as not fitted rather than as a
    # spurious number.
    run = twin_run(LatticeConfig(), seed=42, steps=25, fit_window_steps=25)
    assert run.fit.fitted
    assert run.fit.n_points == 25
    assert run.fit.lam == pytest.approx(0.8625, abs=0.01)
    assert run.fit.r_squared > 0.9
    assert run.fit.lam > 0
    assert not run.fit_check.fitted
    assert run.fit_check.lam is None
    assert max(run.infidelities) < 1e-9
