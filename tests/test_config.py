"""Configuration checks, including the sampling limit of the sinusoid."""

import numpy as np
import pytest

from membrane_arena_3x3.engine import EngineConfig
from membrane_arena_3x3.lattice import LatticeConfig
from membrane_arena_3x3.shields import Observation, YinCalibratorShield


def test_default_schumann_setting_is_below_the_nyquist_limit():
    EngineConfig(schumann=True)  # 7.83 Hz at dt=0.05 -> Nyquist 10 Hz


def test_aliasing_schumann_setting_is_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        EngineConfig(lattice=LatticeConfig(dt=0.1), schumann=True)  # Nyquist 5 Hz


def test_aliasing_setting_is_allowed_when_the_signal_is_off():
    EngineConfig(lattice=LatticeConfig(dt=0.1), schumann=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": -1},
        {"failure_limit": 0},
        {"lattice": LatticeConfig(dt=0.0)},
        {"schumann": True, "schumann_hz": 0.0},
    ],
)
def test_nonsense_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)


def test_sampled_signal_is_a_clipped_sinusoid_of_the_stated_period():
    shield = YinCalibratorShield(max_signal=1.0)
    obs = Observation(1, 0.85, 0.15, 0.20, 0.5, 0.5, 0.0)  # zero feedback
    values = [
        shield.propose(obs, k, True, 7.83, 0.02, 0.05).schumann for k in range(200)
    ]
    expected = [0.02 * np.sin(2 * np.pi * 7.83 * k * 0.05) for k in range(200)]
    assert np.allclose(values, expected)
    assert max(values) <= 0.02 + 1e-12
