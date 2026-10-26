"""Tests for the true phase damping experiment (SYNTHETIC; not part of the package)."""

import numpy as np
import pytest

from experiments.regimes import REGIMES


def test_off_diagonal_damping_keeps_trace_diagonal_and_positivity():
    from experiments.phase_damping import off_diagonal_damping
    from membrane_arena_3x3.core import is_valid_density_matrix, random_density_matrix

    rng = np.random.default_rng(0)
    rho = random_density_matrix(rng, 32)
    out = off_diagonal_damping(rho, 0.3)
    assert np.allclose(np.diag(out), np.diag(rho))
    assert np.trace(out).real == pytest.approx(1.0)
    assert is_valid_density_matrix(out)
    assert np.trace(out @ out).real < np.trace(rho @ rho).real  # decoherence lowers purity
    assert off_diagonal_damping(rho, 0.0) is rho


def test_published_dephasing_is_not_trace_free_phase_damping():
    from membrane_arena_3x3.core import apply_dephasing, random_density_matrix

    rng = np.random.default_rng(1)
    rho = random_density_matrix(rng, 32)
    out = apply_dephasing(rho, 0.05)
    assert not np.allclose(np.diag(out), np.diag(rho))  # populations move: not phase damping


def test_term_decomposition_reproduces_the_published_step():
    from experiments.phase_damping import TERMS, candidate_from_terms
    from membrane_arena_3x3.engine import MembraneArena3x3

    eng = MembraneArena3x3(REGIMES["stock"].engine_config(42))
    eng.step()
    signal = eng.calibrations[-1]["total_signal"]
    ref = eng.lattice.propose_step(signal)
    mine = candidate_from_terms(eng.lattice, signal, {t: True for t in TERMS}, True)
    assert np.allclose(ref, mine, atol=1e-12)


def test_phase_damped_arena_shares_initial_states_with_the_published_model():
    from experiments.phase_damping import phase_damped_arena
    from membrane_arena_3x3.engine import MembraneArena3x3

    cfg = REGIMES["stock"].engine_config(5)
    a = MembraneArena3x3(cfg)
    b = phase_damped_arena(cfg)
    assert np.array_equal(a.lattice.state, b.lattice.state)
    assert type(a.lattice).__name__ == "MembraneLattice"
    assert b.run()["steps_executed"] >= 1
