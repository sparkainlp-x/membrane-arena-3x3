"""Density-matrix primitives: Hermiticity, unit trace, positive semidefiniteness."""

import numpy as np
import pytest

from membrane_arena_3x3.core import (
    apply_dephasing,
    crystallization_mirror,
    fidelity,
    is_valid_density_matrix,
    normalized_entropy,
    project_valid,
    purity,
    random_density_matrix,
)


def assert_density_matrix(rho, tol=1e-10):
    assert np.allclose(rho, rho.conj().T, atol=tol)
    assert abs(np.trace(rho).real - 1.0) < tol
    assert np.linalg.eigvalsh(rho).min() >= -tol


@pytest.mark.parametrize("seed", range(5))
def test_random_density_matrix_is_valid(seed):
    rho = random_density_matrix(np.random.default_rng(seed))
    assert_density_matrix(rho)
    assert is_valid_density_matrix(rho)
    assert 1 / 32 <= purity(rho) <= 1.0
    assert 0.0 <= normalized_entropy(rho) <= 1.0 + 1e-9


def test_project_valid_repairs_a_bad_matrix():
    rng = np.random.default_rng(0)
    bad = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    assert not is_valid_density_matrix(bad)
    assert_density_matrix(project_valid(bad))


def test_project_valid_falls_back_to_the_maximally_mixed_state():
    rho = project_valid(np.zeros((4, 4), dtype=complex))
    assert np.allclose(rho, np.eye(4) / 4)


@pytest.mark.parametrize("make_bad", [
    lambda r: r * 1.1,  # trace 1.1
    lambda r: r + 1e-3 * np.triu(np.ones_like(r), 1),  # not Hermitian
    lambda r: r + np.diag(np.r_[-1.0, np.zeros(len(r) - 1)]),  # negative eigenvalue
    lambda r: np.where(np.eye(len(r)) == 1, np.nan, r),  # non-finite
])
def test_validity_check_rejects_bad_matrices(make_bad):
    rho = random_density_matrix(np.random.default_rng(1), 6)
    assert is_valid_density_matrix(rho)
    assert not is_valid_density_matrix(make_bad(rho))


def test_validity_check_rejects_non_square():
    assert not is_valid_density_matrix(np.ones((2, 3)))


@pytest.mark.parametrize("seed", range(3))
def test_dephasing_and_mirror_keep_the_state_valid(seed):
    rho = random_density_matrix(np.random.default_rng(seed))
    for gamma in (0.0, 0.025, 1.0):
        assert_density_matrix(apply_dephasing(rho, gamma))
    for strength in (0.0, 0.1, 0.5):
        assert_density_matrix(crystallization_mirror(rho, strength))


def test_mirror_raises_purity():
    rho = random_density_matrix(np.random.default_rng(0))
    assert purity(crystallization_mirror(rho, 0.5)) > purity(rho)


def test_fidelity_is_bounded_symmetric_and_one_on_identical_states():
    rng = np.random.default_rng(2)
    a, b = random_density_matrix(rng), random_density_matrix(rng)
    assert fidelity(a, a) == pytest.approx(1.0, abs=1e-9)
    assert fidelity(a, b) == pytest.approx(fidelity(b, a), abs=1e-9)
    assert 0.0 <= fidelity(a, b) <= 1.0
