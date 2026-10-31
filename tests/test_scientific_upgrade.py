"""Scientific upgrade: configurable guards, adaptive retry, metric names,
trace-preserving operators, and benchmark determinism. All SYNTHETIC."""

import numpy as np
import pytest

from membrane_arena_3x3.core import (
    apply_phase_damping_tp,
    exact_unitary_step,
    is_valid_density_matrix,
    max_off_diagonal_coherence,
    mean_off_diagonal_coherence,
    purity,
    random_density_matrix,
)
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.lattice import LatticeConfig, MembraneLattice
from membrane_arena_3x3.shields import GuardConfig, YangObserverShield


def test_guard_thresholds_are_configurable_and_logged():
    guards = GuardConfig(
        residual_target=0.5,
        divergence_min=0.0,
        divergence_max=0.9,
        purity_floor=0.01,
        entropy_target=0.7,
    )
    result = MembraneArena3x3(
        EngineConfig(steps=6, seed=42, guards=guards, failure_limit=7)
    ).run()
    assert result["guards"] == {
        "residual_target": 0.5,
        "divergence_min": 0.0,
        "divergence_max": 0.9,
        "purity_floor": 0.01,
        "entropy_target": 0.7,
        "failure_limit": 7,
    }


def test_divergence_min_is_now_a_real_check():
    # A floor above any possible mean divergence rejects, once the
    # other guards are relaxed so they do not fail first (guard
    # priority is residual, then divergence max, then purity, then
    # divergence min — the same order as the stock is_healthy).
    guards = GuardConfig(
        residual_target=1.0,
        divergence_max=1.0,
        purity_floor=0.0,
        divergence_min=0.999,
    )
    result = MembraneArena3x3(
        EngineConfig(steps=6, seed=42, guards=guards)
    ).run()
    reasons = {r["reason"] for r in result["rejections"]}
    assert "divergence_below_minimum" in reasons


def test_rejection_telemetry_names_guard_cells_and_metrics():
    arena = MembraneArena3x3(EngineConfig(seed=42))
    arena.run()
    assert arena.rejections, "default seed-42 run must reject at least once"
    first = arena.rejections[0]
    assert first["failed_guard"] == first["reason"]
    assert first["offending_cells"], "offending cells must be listed"
    assert "max_residual" in first["metrics"]
    assert first["thresholds"]["residual_target"] == 0.20
    assert first["metrics"]["max_residual"] > 0.20


def test_adaptive_retry_beats_no_retry_on_seed_42():
    with_retry = MembraneArena3x3(EngineConfig(seed=42)).run()
    without = MembraneArena3x3(EngineConfig(seed=42, adaptive_retry=False)).run()
    assert with_retry["accepted_steps"] > without["accepted_steps"]
    assert with_retry["repair_accepts"] > 0
    assert without["repair_accepts"] == 0


def test_a_rejected_step_counts_one_failure_after_distinct_attempts():
    arena = MembraneArena3x3(EngineConfig(seed=42))
    arena.run()
    assert arena.rejections
    for rejection in arena.rejections:
        dts = [a["dt"] for a in rejection["repair_attempts"]]
        assert len(dts) == len(set(dts)), "repair attempts must not repeat dt"
        assert dts == sorted(dts, reverse=True), "repair dt must shrink"
    # Five counted failures latch the breaker, not five identical retries.
    assert arena.failures == 5
    assert len(arena.rejections) == 5


def test_metric_names_are_purity_and_stability_penalty():
    arena = MembraneArena3x3(EngineConfig(seed=3))
    observation = arena.observer.observe(
        arena.lattice.state, arena.lattice.reference, 0
    )
    d = observation.to_dict()
    assert "mean_purity" in d and "stability_penalty" in d
    assert "mean_coherence" not in d and "lyapunov_proxy" not in d
    assert "mean_off_diagonal_coherence" in d
    assert 0.0 <= d["mean_off_diagonal_coherence"] <= 1.0
    # Old attribute access still works for callers that read it.
    assert observation.mean_coherence == observation.mean_purity
    assert observation.lyapunov_proxy == observation.stability_penalty


def test_off_diagonal_coherence_is_zero_for_diagonal_states():
    rho = np.diag([0.5, 0.3, 0.2]).astype(complex)
    assert mean_off_diagonal_coherence(rho) == 0.0
    assert max_off_diagonal_coherence(rho) == 0.0
    rho2 = np.array([[0.5, 0.4], [0.4, 0.5]], dtype=complex)
    assert mean_off_diagonal_coherence(rho2) == pytest.approx(0.4)


def test_trace_preserving_phase_damping_keeps_diagonal_and_trace():
    rng = np.random.default_rng(0)
    rho = random_density_matrix(rng, 8)
    out = apply_phase_damping_tp(rho, 0.5)
    assert np.allclose(np.diag(out), np.diag(rho))
    assert abs(np.trace(out).real - 1.0) < 1e-12
    assert is_valid_density_matrix(out)
    assert mean_off_diagonal_coherence(out) < mean_off_diagonal_coherence(rho)


def test_exact_unitary_stage_preserves_trace_and_purity():
    rng = np.random.default_rng(1)
    rho = random_density_matrix(rng, 8)
    h = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    h = (h + h.conj().T) / 2
    out = exact_unitary_step(rho, h, 0.05)
    assert abs(np.trace(out).real - 1.0) < 1e-9
    assert purity(out) == pytest.approx(purity(rho), abs=1e-9)
    assert is_valid_density_matrix(out)


def test_evolution_stages_are_independently_testable():
    lattice = MembraneLattice(
        LatticeConfig(operator_mode="exact"), np.random.default_rng(2)
    )
    rho = lattice.state[0, 0]
    h = lattice.hamiltonians[0, 0]
    assert is_valid_density_matrix(lattice.unitary_stage(rho, h, 0.05))
    assert is_valid_density_matrix(lattice.dissipative_stage(rho, 0.025))
    assert lattice.mirror_stage(rho, 0.1).shape == rho.shape
    delta = lattice.interaction_delta(0, 0)
    assert abs(np.trace(delta).real) < 1e-9  # interaction is trace-free


def test_exact_mode_changes_the_failure_guard_on_seed_42():
    # Measured: legacy first fails on residual, exact first fails on purity.
    legacy = MembraneArena3x3(EngineConfig(seed=42)).run()
    exact = MembraneArena3x3(
        EngineConfig(
            seed=42,
            lattice=LatticeConfig(operator_mode="exact"),
        )
    ).run()
    assert legacy["rejections"][0]["reason"] == "residual_above_target"
    assert exact["rejections"][0]["reason"] == "purity_too_low"
