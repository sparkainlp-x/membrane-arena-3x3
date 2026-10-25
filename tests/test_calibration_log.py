"""The calibration log records bounded signals and the parameters in effect."""

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.lattice import LatticeConfig
from membrane_arena_3x3.shields import Observation, YinCalibratorShield


def test_log_records_parameters_in_effect_and_they_never_change():
    lattice = LatticeConfig(coupling=0.03, dephasing=0.4, mirror_strength=0.2)
    result = MembraneArena3x3(EngineConfig(lattice=lattice, steps=4)).run()
    for entry in result["calibrations"]:
        assert (entry["coupling"], entry["dephasing"], entry["mirror_strength"]) == (
            0.03,
            0.4,
            0.2,
        )


def test_total_signal_is_always_within_the_bound():
    shield = YinCalibratorShield(max_signal=0.05)
    extreme = Observation(
        step=1,
        mean_coherence=0.0,
        max_divergence=1.0,
        max_residual=5.0,
        min_purity=0.0,
        mean_entropy=1.0,
        lyapunov_proxy=9.0,
    )
    for step in range(50):
        action = shield.propose(extreme, step, True, 7.83, 10.0, 0.05)
        assert abs(action.feedback) <= 0.05
        assert abs(action.schumann) <= 0.05
        assert abs(action.total_signal) <= 0.05


def test_default_feedback_is_saturated_at_the_clip():
    # Documented limit: the coherence target (0.85) is far above the purity of
    # random 32-dimensional states (about 0.06), so the feedback sits at the clip.
    result = MembraneArena3x3(EngineConfig(steps=6, seed=42)).run()
    assert {entry["feedback"] for entry in result["calibrations"]} == {0.05}
