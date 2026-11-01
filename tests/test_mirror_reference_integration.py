"""Integration contract for the open PR #10 mirror-reference dependency."""

import numpy as np
import pytest

mirror = pytest.importorskip(
    "mirror_reference",
    reason="rate-based mirror reference is pending open PR #10",
)


def test_rate_mapping_and_composition_match_reference_timestep():
    rho = np.diag([0.6, 0.3, 0.1]).astype(complex)
    strength, reference_dt = 0.1, 0.05
    alpha = mirror.rate_from_legacy_strength(strength, reference_dt)
    np.testing.assert_allclose(
        mirror.mirror_flow(rho, alpha, reference_dt),
        mirror.mirror_power(rho, 1 + strength),
        atol=2e-12,
        rtol=0,
    )
    whole = mirror.mirror_flow(rho, alpha, 0.2)
    parts = mirror.mirror_flow(
        mirror.mirror_flow(rho, alpha, 0.07), alpha, 0.13
    )
    np.testing.assert_allclose(whole, parts, atol=2e-12, rtol=0)


def test_rate_rhs_and_entropy_derivative_match_finite_differences():
    rho = np.diag([0.7, 0.2, 0.1]).astype(complex)
    alpha, dt = 0.4, 1e-6
    difference = (mirror.mirror_flow(rho, alpha, dt) - rho) / dt
    np.testing.assert_allclose(
        difference, mirror.mirror_rhs(rho, alpha), atol=2e-7, rtol=0
    )
    entropy_difference = (
        mirror.entropy(mirror.mirror_flow(rho, alpha, dt))
        - mirror.entropy(rho)
    ) / dt
    assert abs(entropy_difference - mirror.entropy_rate(rho, alpha)) < 2e-6
