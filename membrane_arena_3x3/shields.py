"""Independent Yang Observer and Yin Calibrator shield membranes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .core import fidelity, normalized_entropy, purity, residual_infinity

@dataclass
class Observation:
    """Read-only stability telemetry produced by the Yang Observer Shield."""

    step: int
    mean_coherence: float
    max_divergence: float
    max_residual: float
    min_purity: float
    mean_entropy: float
    lyapunov_proxy: float

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class CalibrationAction:
    """A bounded, fully logged calibrator proposal."""

    step: int
    feedback: float
    schumann: float
    total_signal: float
    coupling: float
    dephasing: float
    mirror_strength: float
    accepted: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

class YangObserverShield:
    """Read-only observer. It never mutates lattice state."""

    def __init__(
        self,
        residual_target: float = 0.20,
        divergence_min: float = 0.02,
        divergence_max: float = 0.60,
    ) -> None:
        self.residual_target = residual_target
        self.divergence_min = divergence_min
        self.divergence_max = divergence_max

    def observe(
        self, state: np.ndarray, reference: np.ndarray, step: int
    ) -> Observation:
        """Compute stability metrics for all nine membrane cells."""
        cells = state.reshape(-1, state.shape[-2], state.shape[-1])
        refs = reference.reshape(-1, reference.shape[-2], reference.shape[-1])

        coherences = []
        divergences = []
        residuals = []
        purities = []
        entropies = []

        for k, rho in enumerate(cells):
            coherences.append(purity(rho))
            residuals.append(residual_infinity(rho, refs[k]))
            purities.append(purity(rho))
            entropies.append(normalized_entropy(rho))

        for a in range(len(cells)):
            for b in range(a + 1, len(cells)):
                divergences.append(1.0 - fidelity(cells[a], cells[b]))

        mean_coherence = float(np.mean(coherences))
        max_divergence = float(np.max(divergences))
        max_residual = float(np.max(residuals))
        min_purity = float(np.min(purities))
        mean_entropy = float(np.mean(entropies))

        lyapunov = (
            1.0 * max_residual
            + 0.5 * max(0.0, max_divergence - self.divergence_max)
            + 0.5 * abs(mean_entropy - 0.5)
            + 0.5 * (1.0 - min_purity)
        )

        return Observation(
            step=step,
            mean_coherence=mean_coherence,
            max_divergence=max_divergence,
            max_residual=max_residual,
            min_purity=min_purity,
            mean_entropy=mean_entropy,
            lyapunov_proxy=float(lyapunov),
        )

    def is_healthy(self, observation: Observation) -> tuple[bool, str]:
        """Return whether the observation passes stability guardrails."""
        if observation.max_residual > self.residual_target:
            return False, "residual_above_target"
        if observation.max_divergence > self.divergence_max:
            return False, "divergence_above_maximum"
        if observation.min_purity < 0.05:
            return False, "purity_too_low"
        return True, "ok"

class YinCalibratorShield:
    """Bounded calibrator. It proposes control but cannot bypass safety checks."""

    def __init__(
        self,
        coherence_target: float = 0.85,
        divergence_target: float = 0.15,
        residual_target: float = 0.20,
        max_signal: float = 0.05,
        gain_coherence: float = 0.8,
        gain_divergence: float = 0.4,
        gain_residual: float = 0.6,
    ) -> None:
        self.coherence_target = coherence_target
        self.divergence_target = divergence_target
        self.residual_target = residual_target
        self.max_signal = max_signal
        self.gain_coherence = gain_coherence
        self.gain_divergence = gain_divergence
        self.gain_residual = gain_residual
        self.log: list[CalibrationAction] = []

    def propose(
        self,
        observation: Observation,
        step: int,
        schumann_enabled: bool,
        schumann_hz: float,
        schumann_amplitude: float,
        dt: float,
    ) -> CalibrationAction:
        """Return a clipped calibration proposal based on observer telemetry."""
        feedback = (
            self.gain_coherence
            * (self.coherence_target - observation.mean_coherence)
            + self.gain_divergence
            * (observation.max_divergence - self.divergence_target)
            + self.gain_residual
            * (observation.max_residual - self.residual_target)
        )
        feedback = float(np.clip(feedback, -self.max_signal, self.max_signal))

        schumann = 0.0
        if schumann_enabled:
            schumann = schumann_amplitude * np.sin(
                2 * np.pi * schumann_hz * step * dt
            )
            schumann = float(
                np.clip(schumann, -self.max_signal, self.max_signal)
            )

        total = float(
            np.clip(feedback + schumann, -self.max_signal, self.max_signal)
        )

        action = CalibrationAction(
            step=step,
            feedback=feedback,
            schumann=schumann,
            total_signal=total,
            coupling=0.0,
            dephasing=0.0,
            mirror_strength=0.0,
            accepted=False,
            reason="proposed",
        )
        self.log.append(action)
        return action

    def mark(self, action: CalibrationAction, accepted: bool, reason: str) -> None:
        """Record the engine’s accept/reject decision for auditability."""
        action.accepted = accepted
        action.reason = reason
