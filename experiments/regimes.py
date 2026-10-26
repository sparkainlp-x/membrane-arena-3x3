"""Parameter regimes used by the injection experiments (SYNTHETIC).

The intermediate regime is the arithmetic midpoint of the stock and evolved
parameters, fixed by this definition before any intermediate-regime result
was looked at.
"""

from __future__ import annotations

from dataclasses import dataclass

from membrane_arena_3x3.engine import EngineConfig
from membrane_arena_3x3.lattice import LatticeConfig

STOCK = dict(coupling=0.02, dephasing=0.5, mirror_strength=0.10, dt=0.05)
EVOLVED = dict(coupling=0.0339, dephasing=0.2871, mirror_strength=0.1725, dt=0.0128)
STOCK_FIRST_FACTOR, EVOLVED_FIRST_FACTOR = 0.5, 0.2555


def _mid(a: float, b: float) -> float:
    return (a + b) / 2.0


@dataclass(frozen=True)
class Regime:
    name: str
    lattice: dict
    first_factor: float
    mirror_variant: float = 0.0

    def engine_config(self, seed: int, steps: int = 200) -> EngineConfig:
        return EngineConfig(steps=steps, seed=seed, lattice=LatticeConfig(**self.lattice))


REGIMES = {
    "stock": Regime("stock", dict(STOCK), STOCK_FIRST_FACTOR),
    "intermediate": Regime(
        "intermediate",
        {k: _mid(STOCK[k], EVOLVED[k]) for k in STOCK},
        _mid(STOCK_FIRST_FACTOR, EVOLVED_FIRST_FACTOR),
    ),
    "evolved": Regime("evolved", dict(EVOLVED), EVOLVED_FIRST_FACTOR),
}
