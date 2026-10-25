"""3x3 lattice of coupled synthetic membrane cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    DIM,
    apply_dephasing,
    crystallization_mirror,
    fidelity,
    project_valid,
    random_density_matrix,
)

@dataclass
class LatticeConfig:
    """Configuration for the nine-cell lattice and its shared interaction rule."""

    dim: int = DIM
    coupling: float = 0.02
    coupling_min: float = 0.0
    coupling_max: float = 0.10
    dephasing: float = 0.5
    dephasing_min: float = 0.0
    dephasing_max: float = 2.0
    mirror_strength: float = 0.10
    mirror_min: float = 0.0
    mirror_max: float = 0.50
    resonance_threshold: float = 0.90
    dt: float = 0.05

class MembraneLattice:
    """Nine density-matrix cells arranged as a 3x3 grid.

    Index convention: `state[i, j]`, where `i` is the arena and `j` is the
    membrane inside that arena.
    """

    def __init__(self, config: LatticeConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.state = np.empty((3, 3, config.dim, config.dim), dtype=complex)
        self.hamiltonians = np.empty_like(self.state)

        for i in range(3):
            for j in range(3):
                self.state[i, j] = random_density_matrix(rng, config.dim)
                h = rng.normal(size=(config.dim, config.dim)) + 1j * rng.normal(
                    size=(config.dim, config.dim)
                )
                self.hamiltonians[i, j] = (h + h.conj().T) / 2

        self.reference = self.state.copy()

    def neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        """Return same-arena neighbors and corresponding cells in other arenas."""
        cells: list[tuple[int, int]] = []

        for jj in range(3):
            if jj != j:
                cells.append((i, jj))

        for ii in range(3):
            if ii != i:
                cells.append((ii, j))

        return cells

    def interaction_delta(self, i: int, j: int) -> np.ndarray:
        """Compute the shared interaction contribution for one membrane cell."""
        cfg = self.config
        rho = self.state[i, j]
        delta = np.zeros_like(rho)

        for a, b in self.neighbors(i, j):
            other = self.state[a, b]
            score = fidelity(rho, other)
            drive = cfg.coupling * (score - cfg.resonance_threshold)
            delta += drive * (other - rho)

        return delta / max(len(self.neighbors(i, j)), 1)

    def propose_step(self, calibrator_signal: float = 0.0) -> np.ndarray:
        """Return a candidate next state without mutating the current lattice."""
        cfg = self.config
        candidate = np.empty_like(self.state)

        for i in range(3):
            for j in range(3):
                rho = self.state[i, j]
                h = self.hamiltonians[i, j]

                commutator = h @ rho - rho @ h
                unitary = -1j * cfg.dt * commutator
                dissipative = apply_dephasing(rho, cfg.dephasing * cfg.dt)

                delta = unitary + (dissipative - rho)
                delta += self.interaction_delta(i, j) * cfg.dt

                if calibrator_signal:
                    delta += calibrator_signal * cfg.dt * (
                        self.reference[i, j] - rho
                    )

                candidate[i, j] = project_valid(rho + delta)
                candidate[i, j] = crystallization_mirror(
                    candidate[i, j], cfg.mirror_strength
                )

        return candidate

    def commit(self, candidate: np.ndarray) -> None:
        """Atomically replace the lattice state with a validated candidate."""
        self.state = candidate.copy()
