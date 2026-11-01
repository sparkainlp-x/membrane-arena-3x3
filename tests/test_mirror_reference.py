import numpy as np
import pytest

from mirror_reference import (
    entropy,
    entropy_rate,
    mirror_flow,
    mirror_power,
    mirror_rhs,
    purity,
    rate_from_legacy_strength,
)


def random_density(seed=42, dimension=8):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


@pytest.mark.parametrize("seed", range(5))
def test_power_composition(seed):
    rho = random_density(seed)
    sequential = mirror_power(mirror_power(rho, 1.3), 1.7)
    direct = mirror_power(rho, 1.3 * 1.7)
    np.testing.assert_allclose(sequential, direct, atol=2e-12, rtol=0)


def test_timestep_partition_independence():
    rho = random_density()
    many_steps = rho.copy()
    for _ in range(40):
        many_steps = mirror_flow(many_steps, alpha=0.4, dt=0.025)
    one_step = mirror_flow(rho, alpha=0.4, dt=1.0)
    np.testing.assert_allclose(many_steps, one_step, atol=2e-12, rtol=0)


def test_legacy_rate_matches_reference_step():
    rho = random_density()
    strength = 0.10
    reference_dt = 0.05
    alpha = rate_from_legacy_strength(strength, reference_dt)
    np.testing.assert_allclose(
        mirror_flow(rho, alpha, reference_dt),
        mirror_power(rho, 1 + strength),
        atol=2e-12,
        rtol=0,
    )


def test_rank_deficient_degenerate_limit():
    rho = np.diag([0.4, 0.4, 0.2, 0.0])
    result = mirror_power(rho, q=1000)
    np.testing.assert_allclose(
        result, np.diag([0.5, 0.5, 0.0, 0.0]), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "rho", [np.diag([1.0, 0.0, 0.0]), np.eye(4) / 4]
)
def test_fixed_points(rho):
    result = mirror_flow(rho, alpha=0.8, dt=1.0)
    np.testing.assert_allclose(result, rho, atol=1e-12, rtol=0)
    assert abs(entropy_rate(rho, alpha=0.8)) < 1e-12


def test_state_derivative():
    rho = random_density()
    alpha = 0.7
    h = 1e-6
    finite_difference = (mirror_flow(rho, alpha, h) - rho) / h
    np.testing.assert_allclose(
        finite_difference, mirror_rhs(rho, alpha), atol=2e-7, rtol=0
    )


def test_entropy_derivative():
    rho = random_density()
    alpha = 0.7
    h = 1e-6
    finite_difference = (entropy(mirror_flow(rho, alpha, h)) - entropy(rho)) / h
    assert abs(finite_difference - entropy_rate(rho, alpha)) < 2e-6


def test_sharpening_and_validity():
    rho = random_density()
    result = mirror_flow(rho, alpha=0.6, dt=2.0)
    assert entropy(result) <= entropy(rho) + 1e-12
    assert purity(result) >= purity(rho) - 1e-12
    assert np.linalg.eigvalsh(result).min() >= -1e-12
    assert abs(np.trace(result) - 1.0) < 1e-12
    np.testing.assert_allclose(result, result.conj().T, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "rho",
    [
        np.eye(2),
        np.diag([1.2, -0.2]),
        np.array([[1, 1], [0, 0]]),
        np.array([[np.nan]]),
        np.array([0.5, 0.5]),
    ],
)
def test_invalid_states(rho):
    with pytest.raises(ValueError):
        mirror_power(rho, q=1.1)


@pytest.mark.parametrize("q", [0, -1, np.inf, np.nan])
def test_invalid_exponents(q):
    with pytest.raises(ValueError):
        mirror_power(np.eye(2) / 2, q=q)


def test_flow_overflow_is_rejected():
    with pytest.raises(ValueError):
        mirror_flow(np.eye(2) / 2, alpha=1000, dt=1)
