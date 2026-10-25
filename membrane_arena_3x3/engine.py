"""Transactional simulation engine for the 3x3 membrane lattice."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import is_valid_density_matrix
from .lattice import LatticeConfig, MembraneLattice
from .shields import YangObserverShield, YinCalibratorShield

@dataclass
class EngineConfig:
    """Top-level simulation configuration."""

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    steps: int = 200
    seed: int = 42
    schumann: bool = False
    schumann_hz: float = 7.83
    schumann_amplitude: float = 0.02
    failure_limit: int = 5  # consecutive rejected updates before the breaker latches

class MembraneArena3x3:
    """Run a fail-closed 3x3 membrane simulation with independent shields."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.lattice = MembraneLattice(self.config.lattice, self.rng)
        self.observer = YangObserverShield()
        self.calibrator = YinCalibratorShield()
        self.observations: list[dict] = []
        self.calibrations: list[dict] = []
        self.failures = 0
        self.circuit_breaker = False
        self.current_step = 0

    def _validate(self, candidate: np.ndarray) -> tuple[bool, str]:
        """Validate an entire candidate lattice state before commit."""
        cfg = self.config.lattice

        for i in range(3):
            for j in range(3):
                rho = candidate[i, j]
                if not is_valid_density_matrix(rho):
                    return False, f"invalid_density_matrix[{i},{j}]"

        observation = self.observer.observe(
            candidate, self.lattice.reference, self.current_step + 1
        )
        healthy, reason = self.observer.is_healthy(observation)
        if not healthy:
            return False, reason

        if not (
            cfg.coupling_min <= cfg.coupling <= cfg.coupling_max
            and cfg.dephasing_min <= cfg.dephasing <= cfg.dephasing_max
            and cfg.mirror_min <= cfg.mirror_strength <= cfg.mirror_max
        ):
            return False, "calibrator_parameter_out_of_bounds"

        return True, "ok"

    def step(self) -> dict:
        """Execute one transactional global update."""
        if self.circuit_breaker:
            return {
                "status": "circuit_breaker_open",
                "step": self.current_step,
            }

        observation = self.observer.observe(
            self.lattice.state, self.lattice.reference, self.current_step
        )

        action = self.calibrator.propose(
            observation,
            self.current_step,
            self.config.schumann,
            self.config.schumann_hz,
            self.config.schumann_amplitude,
            self.config.lattice.dt,
        )

        # The calibrator never changes these; log the values in effect so the
        # audit trail does not show misleading zeros.
        lattice_cfg = self.config.lattice
        action.coupling = lattice_cfg.coupling
        action.dephasing = lattice_cfg.dephasing
        action.mirror_strength = lattice_cfg.mirror_strength

        candidate = self.lattice.propose_step(action.total_signal)
        accepted, reason = self._validate(candidate)

        if accepted:
            self.lattice.commit(candidate)
            self.failures = 0
        else:
            self.failures += 1
            if self.failures >= self.config.failure_limit:
                self.circuit_breaker = True

        self.calibrator.mark(action, accepted, reason)
        self.calibrations.append(action.to_dict())

        post = self.observer.observe(
            self.lattice.state, self.lattice.reference, self.current_step + 1
        )
        self.observations.append(post.to_dict())
        self.current_step += 1

        return {
            "status": "accepted" if accepted else "rejected",
            "reason": reason,
            "step": self.current_step,
            "observation": post.to_dict(),
            "calibration": action.to_dict(),
        }

    def run(self) -> dict:
        """Run the configured number of steps, stopping if the breaker opens."""
        self.current_step = 0

        for _ in range(self.config.steps):
            result = self.step()
            if result["status"] == "circuit_breaker_open":
                break

        return self.summary()

    def summary(self) -> dict:
        """Return a machine-readable experiment summary."""
        final = self.observations[-1] if self.observations else {}
        accepted = sum(1 for item in self.calibrations if item["accepted"])

        return {
            "status": (
                "circuit_breaker_open"
                if self.circuit_breaker
                else "completed" if self.observations else "no_steps"
            ),
            "steps_requested": self.config.steps,
            "steps_executed": len(self.observations),
            "accepted_steps": accepted,
            "rejected_steps": len(self.calibrations) - accepted,
            "final": final,
            "observations": self.observations,
            "calibrations": self.calibrations,
        }

def run_simulation(config: EngineConfig | None = None) -> dict:
    """Convenience function for running one complete simulation."""
    return MembraneArena3x3(config).run()
