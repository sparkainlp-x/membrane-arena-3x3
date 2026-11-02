"""
Reference implementation of a nonlinear density-matrix mirror flow.

Model
-----
    M_q(rho) = rho**q / trace(rho**q)

For a nonnegative sharpening rate alpha:
    q(dt) = exp(alpha * dt)

Therefore:
    M_q(dt2)(M_q(dt1)(rho)) = M_q(dt1 + dt2)(rho)

The corresponding differential equation is:
    d(rho)/dt = alpha * (
        rho @ log(rho)
        - rho * trace(rho @ log(rho))
    )

This is a synthetic nonlinear model, not a claim of a physical quantum
channel.

"Exact" means the analytical mirror-only flow is evaluated directly.
Floating-point roundoff still applies. Composing this operation with
other dynamics requires a separate numerical convergence study.
"""

from __future__ import annotations

import numpy as np


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return value


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _density_spectrum(
    rho: np.ndarray,
    *,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate a density matrix and return its spectral decomposition.

    Only negative eigenvalues within the absolute tolerance are clipped.
    The accepted spectrum is normalized to sum to one.
    """
    atol = _positive_finite(atol, "atol")
    rho = np.asarray(rho, dtype=np.complex128)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or rho.shape[0] == 0:
        raise ValueError("rho must be a nonempty square matrix")

    if not np.isfinite(rho).all():
        raise ValueError("rho must contain only finite values")

    if not np.allclose(rho, rho.conj().T, atol=atol, rtol=0):
        raise ValueError("rho must be Hermitian")

    if abs(np.trace(rho) - 1.0) > atol:
        raise ValueError("rho must have trace one")

    hermitian = 0.5 * (rho + rho.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)

    if eigenvalues.min() < -atol:
        raise ValueError("rho must be positive semidefinite")

    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalues /= eigenvalues.sum()

    return eigenvalues, eigenvectors


def _reconstruct(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    result = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
    return 0.5 * (result + result.conj().T)


def mirror_power(
    rho: np.ndarray,
    q: float,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Compute rho**q / trace(rho**q), for finite q > 0.

    Uses shifted logarithmic weights to avoid overflow.
    Zero eigenvalues remain zero.
    """
    q = _positive_finite(q, "q")

    eigenvalues, eigenvectors = _density_spectrum(rho, atol=atol)

    support = eigenvalues > 0
    log_values = np.log(eigenvalues[support])

    # Shifted logs are <= 0, so exponentiation cannot overflow.
    shifted_logs = log_values - log_values.max()

    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(q * shifted_logs)

    updated = np.zeros_like(eigenvalues)
    updated[support] = weights / weights.sum()

    return _reconstruct(updated, eigenvectors)


def mirror_flow(
    rho: np.ndarray,
    alpha: float,
    dt: float,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Evaluate the exact mirror-only flow over duration dt (q = exp(alpha*dt)).

    alpha and dt must be finite and nonnegative.
    """
    alpha = _nonnegative_finite(alpha, "alpha")
    dt = _nonnegative_finite(dt, "dt")

    exponent = alpha * dt
    largest_exponent = np.log(np.finfo(np.float64).max)

    if not np.isfinite(exponent) or exponent >= largest_exponent:
        raise ValueError("alpha * dt is too large to represent exp(alpha * dt)")

    return mirror_power(rho, q=float(np.exp(exponent)), atol=atol)


def mirror_rhs(
    rho: np.ndarray,
    alpha: float,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Evaluate d(rho)/dt = alpha * (rho log rho - rho Tr[rho log rho]).

    Uses the continuous extension 0 * log(0) = 0.
    """
    alpha = _nonnegative_finite(alpha, "alpha")

    eigenvalues, eigenvectors = _density_spectrum(rho, atol=atol)

    p_log_p = np.zeros_like(eigenvalues)
    support = eigenvalues > 0
    p_log_p[support] = eigenvalues[support] * np.log(eigenvalues[support])

    eigenvalue_rates = alpha * (p_log_p - eigenvalues * p_log_p.sum())

    return _reconstruct(eigenvalue_rates, eigenvectors)


def entropy(rho: np.ndarray, *, atol: float = 1e-12) -> float:
    """Von Neumann entropy in nats: S(rho) = -Tr[rho log(rho)]."""
    eigenvalues, _ = _density_spectrum(rho, atol=atol)
    positive = eigenvalues[eigenvalues > 0]
    return float(-np.dot(positive, np.log(positive)))


def purity(rho: np.ndarray, *, atol: float = 1e-12) -> float:
    """Compute Tr[rho**2]."""
    eigenvalues, _ = _density_spectrum(rho, atol=atol)
    return float(np.dot(eigenvalues, eigenvalues))


def entropy_rate(
    rho: np.ndarray,
    alpha: float,
    *,
    atol: float = 1e-12,
) -> float:
    """Analytical mirror-only entropy derivative: -alpha * Var_p(log p)."""
    alpha = _nonnegative_finite(alpha, "alpha")

    eigenvalues, _ = _density_spectrum(rho, atol=atol)
    positive = eigenvalues[eigenvalues > 0]

    logs = np.log(positive)
    mean_log = np.dot(positive, logs)
    variance = np.dot(positive, (logs - mean_log) ** 2)

    return float(-alpha * variance)


def rate_from_legacy_strength(strength: float, reference_dt: float) -> float:
    """
    Convert a fixed per-step mirror strength into a rate.

        alpha = log(1 + strength) / reference_dt

    Keep alpha fixed when subsequently changing the timestep.
    """
    strength = _nonnegative_finite(strength, "strength")
    reference_dt = _positive_finite(reference_dt, "reference_dt")
    return float(np.log1p(strength) / reference_dt)
