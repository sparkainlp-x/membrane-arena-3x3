"""Pairwise collision guard, ADOPTED as a real guard (default 0.10).

The stock shields only check MEAN pairwise divergence, which is blind to
a single collided pair (mean 0.458 -> 0.436 for a fully collided pair).
``GuardConfig.min_pairwise_divergence`` (default 0.10) rejects a lattice
whose MINIMUM pairwise divergence falls below the threshold, i.e. some
two cells have collided into nearly the same state. Pass None to disable
(opt out). Retry variants are judged against the full-step candidate's
divergence, not their own, so a smaller-dt retry cannot launder a
collided step past the guard; and run() checks the guard against the
INITIAL state, so a run that starts collided is caught at step 0.

MARGIN (``GuardConfig.min_pairwise_margin``, default 0.05): the guard
fires only when the collapse is pair-specific, i.e. the mean divergence
is still far above the minimum (margin above 0.05). In exact mode TP
damping synchronises the WHOLE lattice, so min and mean fall together
(margin <= 0.032 measured across 5 seeds x ~30 steps) and the guard
does not fire on the global synchronisation — the mean-divergence guard
is the detector for that. A planted collision leaves the mean high
(margin 0.30 at step 1, 0.40 for a legacy 0.95 pair), so it is still
caught. A deliberately planted near-colliding SUBGROUP (the 0.85-start
offset sweep in exact mode, margin 0.14-0.17) IS still caught: that
subgroup genuinely is a collision, so such a sweep needs the guard
opted out (None).
"""

import pytest

from membrane_arena_3x3.core import fidelity, project_valid
from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.lattice import LatticeConfig
from membrane_arena_3x3.shields import GuardConfig

ROW0 = [(0, 0), (0, 1), (0, 2)]
PAIR = [(0, 0), (0, 1)]


def _set_identical(arena, cells):
    rho = arena.lattice.state[cells[0]].copy()
    h = arena.lattice.hamiltonians[cells[0]].copy()
    for i, j in cells:
        arena.lattice.state[i, j] = rho.copy()
        arena.lattice.hamiltonians[i, j] = h.copy()
        arena.lattice.reference[i, j] = rho.copy()


def _blend_toward(target, other, a):
    return project_valid((1.0 - a) * other + a * target)


def _set_pair_fidelity(arena, cells, target):
    rho0 = arena.lattice.state[cells[0]].copy()
    rho1 = arena.lattice.state[cells[1]].copy()
    lo, hi = 0.0, 1.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if fidelity(rho0, _blend_toward(rho0, rho1, mid)) < target:
            lo = mid
        else:
            hi = mid
    arena.lattice.state[cells[1]] = _blend_toward(rho0, rho1, hi)


def _relaxed_guards(min_pairwise=None):
    return GuardConfig(
        residual_target=2.0,
        divergence_max=1.0,
        purity_floor=0.0,
        divergence_min=0.0,
        min_pairwise_divergence=min_pairwise,
    )


def test_pairwise_guard_default_is_0_10_and_rejects_bad_values():
    assert GuardConfig().min_pairwise_divergence == 0.10
    assert GuardConfig().min_pairwise_margin == 0.05
    # Explicit None still opts out.
    assert GuardConfig(min_pairwise_divergence=None).min_pairwise_divergence is None
    with pytest.raises(ValueError):
        GuardConfig(min_pairwise_divergence=-0.01)
    with pytest.raises(ValueError):
        GuardConfig(min_pairwise_divergence=1.0)
    with pytest.raises(ValueError):
        GuardConfig(min_pairwise_divergence=1.5)
    with pytest.raises(ValueError):
        GuardConfig(min_pairwise_margin=-0.01)
    with pytest.raises(ValueError):
        GuardConfig(min_pairwise_margin=1.0)


def test_default_pairwise_guard_leaves_stock_run_unchanged():
    """Adopted default 0.10: the stock run is unchanged (12 steps /
    7 accepted), and matches the guard explicitly disabled."""
    default = MembraneArena3x3(EngineConfig(seed=42)).run()
    explicit = MembraneArena3x3(
        EngineConfig(seed=42, guards=GuardConfig(min_pairwise_divergence=None))
    ).run()
    assert explicit["steps_executed"] == default["steps_executed"] == 12
    assert explicit["accepted_steps"] == default["accepted_steps"] == 7
    assert explicit["final"] == default["final"]


def test_observation_carries_min_divergence():
    arena = MembraneArena3x3(EngineConfig(seed=42))
    obs = arena.observer.observe(
        arena.lattice.state, arena.lattice.reference, 0
    )
    # Random 32-dim cells: max pairwise fidelity ~0.6, so min divergence ~0.4.
    assert obs.min_divergence > 0.3
    _set_identical(arena, PAIR)
    obs = arena.observer.observe(
        arena.lattice.state, arena.lattice.reference, 0
    )
    assert obs.min_divergence == pytest.approx(0.0, abs=1e-9)


def test_pairwise_guard_catches_deliberate_collision():
    """Relaxed stock guards + pairwise guard on: a collided pair is rejected."""
    arena = MembraneArena3x3(
        EngineConfig(
            seed=42,
            guards=_relaxed_guards(min_pairwise=0.05),
            failure_limit=10000,
        )
    )
    _set_pair_fidelity(arena, PAIR, 1.0)
    _set_identical(arena, PAIR)  # matched clocks + matched reference
    result = arena.step()
    assert result["status"] == "rejected"
    assert result["reason"] == "pairwise_divergence_below_minimum"


def test_pairwise_guard_does_not_fire_on_stock_run():
    """Guard on at 0.05: the stock run is unchanged, no pairwise rejections."""
    arena = MembraneArena3x3(
        EngineConfig(seed=42, guards=GuardConfig(min_pairwise_divergence=0.05))
    )
    summary = arena.run()
    assert summary["steps_executed"] == 12
    assert summary["accepted_steps"] == 7
    assert all(
        r["reason"] != "pairwise_divergence_below_minimum"
        for r in summary["rejections"]
    )
    assert all(
        obs["min_divergence"] > 0.05 for obs in summary["observations"]
    )


def test_pairwise_guard_catches_three_identical_cells():
    arena = MembraneArena3x3(
        EngineConfig(
            seed=42,
            guards=_relaxed_guards(min_pairwise=0.05),
            failure_limit=10000,
        )
    )
    _set_identical(arena, ROW0)
    result = arena.step()
    assert result["status"] == "rejected"
    assert result["reason"] == "pairwise_divergence_below_minimum"


def _set_pair_same_clock_and_reference(arena, cells):
    """Matched clocks (same H) and matched reference, WITHOUT touching state."""
    h = arena.lattice.hamiltonians[cells[0]].copy()
    for i, j in cells:
        arena.lattice.hamiltonians[i, j] = h.copy()
    arena.lattice.reference[cells[1]] = arena.lattice.state[cells[0]].copy()


def test_pairwise_guard_catches_0_95_collision_at_adopted_threshold():
    """Adopted threshold 0.10: a 0.95 pair is caught on the first step,
    and every adaptive-retry variant is rejected too — not only because
    the variants fail the threshold on their own trajectory, but because
    they are judged against the full-step reference (see
    test_retry_cannot_launder_collision_at_0_05)."""
    arena = MembraneArena3x3(
        EngineConfig(
            seed=42,
            guards=_relaxed_guards(min_pairwise=0.10),
            failure_limit=10000,
        )
    )
    _set_pair_fidelity(arena, PAIR, 0.95)
    _set_pair_same_clock_and_reference(arena, PAIR)
    result = arena.step()
    assert result["status"] == "rejected"
    assert result["reason"] == "pairwise_divergence_below_minimum"
    assert all(
        not attempt["accepted"] for attempt in result["repair_attempts"]
    )


def test_initial_state_collision_caught_at_step_zero():
    """A run that STARTS collided is caught at step 0: the breaker
    latches with 0 accepted steps, before any step is taken."""
    arena = MembraneArena3x3(EngineConfig(seed=42))
    _set_pair_fidelity(arena, PAIR, 0.95)
    _set_pair_same_clock_and_reference(arena, PAIR)
    summary = arena.run()
    assert summary["status"] == "circuit_breaker_open"
    assert summary["steps_executed"] == 0
    assert summary["accepted_steps"] == 0
    assert len(summary["rejections"]) == 1
    first = summary["rejections"][0]
    assert first["step"] == 0
    assert first["reason"] == "pairwise_divergence_below_minimum"


def test_initial_state_just_above_threshold_is_not_caught_at_step_zero():
    """Boundary: a pair just ABOVE the threshold (fidelity 0.89, min
    divergence 0.11) is NOT caught at step 0 — the check is a strict
    less-than. It runs for a step or two first (the pair wobbles: it
    diverges on step 1, converges back on step 2) and is caught on
    pairwise grounds once a candidate would push it below the
    threshold. (Fidelity exactly 0.90 is a floating-point knife-edge
    — the bisection lands a hair above it — so the test uses 0.89.)"""
    arena = MembraneArena3x3(
        EngineConfig(
            seed=42,
            guards=_relaxed_guards(min_pairwise=0.10),
            failure_limit=10000,
            steps=30,
        )
    )
    _set_pair_fidelity(arena, PAIR, 0.89)
    _set_pair_same_clock_and_reference(arena, PAIR)
    summary = arena.run()
    assert summary["accepted_steps"] >= 1
    assert all(r["step"] != 0 for r in summary["rejections"])
    assert any(
        r["reason"] == "pairwise_divergence_below_minimum"
        for r in summary["rejections"]
    )


def test_retry_cannot_launder_collision_at_0_05():
    """The laundering hole, closed structurally. At threshold 0.05 the
    full step of a 0.95 pair is rejected (min divergence 0.0475), but
    the dt x0.5 retry moves the pair apart (fidelity 0.95 -> 0.9457,
    min divergence 0.0543) and used to be accepted. Retry variants are
    now judged against the full-step reference, so the retry is
    rejected too — at ANY threshold, not only at 0.10."""
    arena = MembraneArena3x3(
        EngineConfig(
            seed=42,
            guards=_relaxed_guards(min_pairwise=0.05),
            failure_limit=10000,
        )
    )
    _set_pair_fidelity(arena, PAIR, 0.95)
    _set_pair_same_clock_and_reference(arena, PAIR)
    # The retry variant, judged on its OWN trajectory, would pass.
    # Reproduce the engine's first retry variant exactly: the real
    # calibrator signal, dt x0.5, mirror x0.5, coupling x0.5.
    cfg = arena.config.lattice
    obs0 = arena.observer.observe(arena.lattice.state, arena.lattice.reference, 0)
    action = arena.calibrator.propose(obs0, 0, False, 7.83, 0.02, cfg.dt)
    variant = arena.lattice.propose_step(
        action.total_signal,
        dt=cfg.dt * 0.5,
        mirror_strength=cfg.mirror_strength * 0.5,
        coupling=cfg.coupling * 0.5,
    )
    own = arena.observer.observe(variant, arena.lattice.reference, 1)
    assert own.min_divergence > 0.05
    # ...but the engine step rejects it anyway, by the full-step reference.
    result = arena.step()
    assert result["status"] == "rejected"
    assert result["reason"] == "pairwise_divergence_below_minimum"
    assert result["pairwise_reference_ok"] is False
    assert result["pairwise_reference_min_divergence"] < 0.05
    assert result["repair_attempts"], "retries must be attempted and logged"
    assert all(
        not attempt["accepted"] for attempt in result["repair_attempts"]
    )
    assert all(
        attempt["pairwise_reference_ok"] is False
        for attempt in result["repair_attempts"]
    )


def test_margin_guard_does_not_fire_on_exact_global_synchronisation():
    """The resolution of the guard-vs-exact interaction. In exact mode
    TP damping synchronises the whole lattice: min and mean divergence
    fall together (margin <= 0.032 measured across 5 seeds), so the
    pairwise guard — which now requires a margin above 0.05 — does not
    fire. The exact control runs to the MEAN collapse (~30 accepted
    steps), exactly as with the guard opted out. Before the margin,
    the same run was cut at ~step 9 by the pairwise guard."""
    for seed in (42, 1, 7):
        default = MembraneArena3x3(
            EngineConfig(seed=seed, lattice=LatticeConfig(operator_mode="exact"))
        ).run()
        opted_out = MembraneArena3x3(
            EngineConfig(
                seed=seed,
                guards=GuardConfig(min_pairwise_divergence=None),
                lattice=LatticeConfig(operator_mode="exact"),
            )
        ).run()
        assert default["accepted_steps"] == opted_out["accepted_steps"]
        assert default["rejections"][0]["reason"] == "divergence_below_minimum"
        assert default["accepted_steps"] >= 28


def test_margin_guard_still_catches_planted_exact_collision():
    """A planted 1.0 collision in exact mode leaves the mean high
    (margin 0.30 at step 1), so the margin guard still catches it at
    step 0 — the margin only exempts lattice-wide synchronisation."""
    arena = MembraneArena3x3(
        EngineConfig(seed=42, lattice=LatticeConfig(operator_mode="exact"))
    )
    _set_identical(arena, PAIR)
    _set_pair_same_clock_and_reference(arena, PAIR)
    summary = arena.run()
    assert summary["status"] == "circuit_breaker_open"
    assert summary["steps_executed"] == 0
    assert summary["accepted_steps"] == 0
    assert summary["rejections"][0]["reason"] == "pairwise_divergence_below_minimum"


def test_margin_guard_catches_planted_near_colliding_subgroup():
    """A deliberately planted near-colliding SUBGROUP in exact mode
    (the 0.85-start offset sweep: three cells at fidelity 0.85 with
    matched clocks) has a large margin (0.14-0.17) once it converges
    past 0.90, so the margin guard DOES catch it — that subgroup
    genuinely is a collision. Such a sweep needs the guard opted out
    (None); the guard is not the tool for studying a planted
    near-collision in exact mode."""
    arena = MembraneArena3x3(
        EngineConfig(
            seed=42,
            lattice=LatticeConfig(
                operator_mode="exact",
                dt_multipliers=(1.0, 2.0, 2.0) + (1.0,) * 6,
            ),
        )
    )
    for other in ROW0[1:]:
        _set_pair_fidelity(arena, [ROW0[0], other], 0.85)
    h = arena.lattice.hamiltonians[ROW0[0]].copy()
    for i, j in ROW0:
        arena.lattice.hamiltonians[i, j] = h.copy()
    summary = arena.run()
    assert summary["rejections"][0]["reason"] == "pairwise_divergence_below_minimum"
    assert summary["accepted_steps"] <= 1


def test_pairwise_collapsed_is_false_when_margin_is_small():
    """Direct unit check of the margin condition: on an exact-control
    observation whose min divergence is already below the threshold
    (0.0998 < 0.10 at step 9) but whose margin is small (0.0118),
    the guard does not fire."""
    arena = MembraneArena3x3(
        EngineConfig(
            seed=42,
            guards=GuardConfig(min_pairwise_divergence=None),
            lattice=LatticeConfig(operator_mode="exact"),
        )
    )
    for _ in range(9):
        arena.step()
    obs = arena.observations[-1]
    assert obs["min_divergence"] < 0.10
    assert obs["mean_divergence"] - obs["min_divergence"] < 0.05
    assert arena.observer.pairwise_collapsed_values(
        obs["min_divergence"], obs["mean_divergence"]
    ) is False
