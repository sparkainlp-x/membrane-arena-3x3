"""Transactional simulation engine for the 3x3 membrane lattice."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import platform
from pathlib import Path
import subprocess

import numpy as np

from .core import is_valid_density_matrix
from .lattice import LatticeConfig, MembraneLattice
from .shields import GuardConfig, YangObserverShield, YinCalibratorShield


@dataclass
class EngineConfig:
    """Top-level simulation configuration."""

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    steps: int = 200
    seed: int = 42
    schumann: bool = False
    schumann_hz: float = 7.83
    schumann_amplitude: float = 0.02
    failure_limit: int = 5  # consecutive rejected steps before the breaker latches
    # Bounded adaptive retry (default on): on a rejection, try variants
    # with a smaller dt, a softened mirror and a reduced coupling. One
    # failure is counted only after all repair attempts fail, so a run
    # can no longer stop from repeated identical rejections.
    adaptive_retry: bool = True
    max_retries: int = 3
    retry_dt_factors: tuple[float, ...] = (0.5, 0.25, 0.125)
    retry_mirror_factors: tuple[float, ...] = (0.5, 0.0, 0.0)
    retry_coupling_factors: tuple[float, ...] = (0.5, 0.25, 0.0)
    retry_policy: str = "legacy_model_changing"
    drive_time_mode: str = "legacy_step_index"

    def __post_init__(self) -> None:
        """Reject settings that make a run meaningless or alias the signal."""
        if self.steps < 0:
            raise ValueError("steps must be >= 0")
        if self.failure_limit < 1:
            raise ValueError("failure_limit must be >= 1")
        if not self.lattice.dt > 0:
            raise ValueError("dt must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_policy not in ("legacy_model_changing", "dt_only"):
            raise ValueError("retry_policy must be 'legacy_model_changing' or 'dt_only'")
        if self.drive_time_mode not in ("legacy_step_index", "accepted_time"):
            raise ValueError("drive_time_mode must be 'legacy_step_index' or 'accepted_time'")
        if len(self.retry_dt_factors) < self.max_retries:
            raise ValueError("retry_dt_factors must cover max_retries")
        if len(self.retry_mirror_factors) < self.max_retries:
            raise ValueError("retry_mirror_factors must cover max_retries")
        if len(self.retry_coupling_factors) < self.max_retries:
            raise ValueError("retry_coupling_factors must cover max_retries")
        for factor in self.retry_dt_factors:
            if not 0.0 < factor < 1.0:
                raise ValueError("retry_dt_factors must be in (0, 1)")
        for factor in (*self.retry_mirror_factors, *self.retry_coupling_factors):
            if not 0.0 <= factor <= 1.0:
                raise ValueError("retry mirror/coupling factors must be in [0, 1]")
        if self.schumann:
            if not self.schumann_hz > 0:
                raise ValueError("schumann_hz must be > 0")
            nyquist = 0.5 / self.lattice.dt
            if self.schumann_hz >= nyquist:
                raise ValueError(
                    f"schumann_hz={self.schumann_hz} is at or above the sampling "
                    f"Nyquist limit {nyquist:g} Hz for dt={self.lattice.dt}; "
                    "the sampled sinusoid would alias"
                )

    def guards_dict(self) -> dict:
        """All guard thresholds, including failure_limit, for JSON output."""
        out = self.guards.to_dict()
        out["failure_limit"] = self.failure_limit
        return out


class MembraneArena3x3:
    """Run a fail-closed 3x3 membrane simulation with independent shields."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.lattice = MembraneLattice(self.config.lattice, self.rng)
        self.observer = YangObserverShield(guard=self.config.guards)
        self.calibrator = YinCalibratorShield()
        self.observations: list[dict] = []
        self.calibrations: list[dict] = []
        self.rejections: list[dict] = []
        self.repair_attempts_total = 0
        self.repair_accepts = 0
        self.failures = 0
        self.circuit_breaker = False
        self.current_step = 0
        self.accepted_time = 0.0
        self.attempt_ledger: list[dict] = []

    def _validate_detailed(
        self, candidate: np.ndarray, effective: dict | None = None
    ) -> tuple[bool, str, dict]:
        """Validate a candidate without observing invalid matrix data."""
        cfg = self.config.lattice
        effective = effective or {
            "dt": cfg.dt,
            "coupling": cfg.coupling,
            "dephasing": cfg.dephasing,
            "mirror_strength": cfg.mirror_strength,
        }
        try:
            candidate = np.asarray(candidate)
        except (TypeError, ValueError):
            candidate = np.asarray([])
        shape = candidate.shape
        if shape != self.lattice.state.shape:
            reason = "invalid_candidate_shape"
            return False, reason, {
                "failed_guard": reason,
                "guard_violations": [reason],
                "offending_cells": ["lattice:shape"],
                "offending_pairs": [],
                "offending_by_guard": {reason: ["lattice:shape"]},
                "metrics": {},
                "thresholds": self._thresholds(),
            }
        if candidate.dtype.kind not in "fciu":
            reason = "invalid_candidate_dtype"
            return False, reason, {
                "failed_guard": reason,
                "guard_violations": [reason],
                "offending_cells": ["lattice:dtype"],
                "offending_pairs": [],
                "offending_by_guard": {reason: ["lattice:dtype"]},
                "metrics": {},
                "thresholds": self._thresholds(),
            }

        invalid_cells: list[str] = []
        for i in range(3):
            for j in range(3):
                rho = candidate[i, j]
                if not is_valid_density_matrix(rho):
                    invalid_cells.append(f"{i},{j}")
        if invalid_cells:
            reason = f"invalid_density_matrix[{invalid_cells[0]}]"
            return False, reason, {
                "failed_guard": reason,
                "guard_violations": ["invalid_density_matrix"],
                "offending_cells": invalid_cells,
                "offending_pairs": [],
                "offending_by_guard": {"invalid_density_matrix": invalid_cells},
                "metrics": {},
                "thresholds": self._thresholds(),
            }

        observation = self.observer.observe(
            candidate, self.lattice.reference, self.current_step + 1
        )
        report = self.observer.guard_report(observation)
        violations = dict(report["offending_by_guard"])

        configured_violations = []
        effective_violations = []
        if not np.isfinite(effective["dt"]) or effective["dt"] <= 0:
            effective_violations.append("dt")
        for name, low, high in (
            ("coupling", cfg.coupling_min, cfg.coupling_max),
            ("dephasing", cfg.dephasing_min, cfg.dephasing_max),
            ("mirror_strength", cfg.mirror_min, cfg.mirror_max),
        ):
            if not np.isfinite(getattr(cfg, name)) or not low <= getattr(cfg, name) <= high:
                configured_violations.append(name)
            value = effective[name]
            if not np.isfinite(value) or not low <= value <= high:
                effective_violations.append(name)
        if configured_violations:
            violations["calibrator_parameter_out_of_bounds"] = [
                f"lattice:configured_{name}" for name in configured_violations
            ]
        if effective_violations:
            violations["effective_parameter_out_of_bounds"] = [
                f"lattice:effective_{name}" for name in effective_violations
            ]

        reason = "ok"
        healthy, observer_reason = self.observer.is_healthy(observation)
        if not healthy:
            reason = observer_reason
        elif configured_violations:
            reason = "calibrator_parameter_out_of_bounds"
        elif effective_violations:
            reason = "effective_parameter_out_of_bounds"
        return reason == "ok", reason, {
            **report,
            "failed_guard": None if reason == "ok" else reason,
            "guard_violations": [name for name, cells in violations.items() if cells],
            "offending_by_guard": violations,
            "offending_cells": list(dict.fromkeys(
                item for cells in violations.values() for item in cells
                if "-" not in item and not item.startswith("lattice:")
            )),
            "offending_pairs": [
                item for cells in violations.values() for item in cells if "-" in item
            ],
            "thresholds": {
                **report["thresholds"],
                "failure_limit": self.config.failure_limit,
                "parameter_bounds": {
                    "dt": {"min_exclusive": 0.0},
                    "coupling": [cfg.coupling_min, cfg.coupling_max],
                    "dephasing": [cfg.dephasing_min, cfg.dephasing_max],
                    "mirror_strength": [cfg.mirror_min, cfg.mirror_max],
                },
            },
        }

    def _validate(self, candidate: np.ndarray) -> tuple[bool, str]:
        """Validate an entire candidate lattice state before commit."""
        ok, reason, _ = self._validate_detailed(candidate)
        return ok, reason

    def _validate_with_telemetry(
        self, candidate: np.ndarray, effective: dict
    ) -> tuple[bool, str, dict]:
        """Preserve validation overrides while retaining fail-closed checks."""
        validator = self._validate
        is_custom = (
            getattr(validator, "__func__", None) is not MembraneArena3x3._validate
            or "_validate" in self.__dict__
        )
        detailed_ok, detailed_reason, report = self._validate_detailed(
            candidate, effective
        )
        if not is_custom:
            return detailed_ok, detailed_reason, report

        custom_ok, custom_reason = validator(candidate)
        hard_violations = {
            "invalid_candidate_shape",
            "invalid_candidate_dtype",
            "invalid_density_matrix",
            "calibrator_parameter_out_of_bounds",
            "effective_parameter_out_of_bounds",
        }
        unsafe = hard_violations.intersection(report["guard_violations"])
        accepted = bool(custom_ok) and not unsafe
        reason = (
            detailed_reason if unsafe else custom_reason
        )
        if not accepted:
            report["failed_guard"] = reason
            report["guard_violations"] = list(dict.fromkeys(
                [*report["guard_violations"], reason]
            ))
            report["offending_by_guard"].setdefault(reason, [])
        return accepted, reason, report

    def _thresholds(self) -> dict:
        cfg = self.config.lattice
        return {
            **self.config.guards_dict(),
            "parameter_bounds": {
                "dt": {"min_exclusive": 0.0},
                "coupling": [cfg.coupling_min, cfg.coupling_max],
                "dephasing": [cfg.dephasing_min, cfg.dephasing_max],
                "mirror_strength": [cfg.mirror_min, cfg.mirror_max],
            },
        }

    def _provenance(self) -> dict:
        root = Path(__file__).resolve().parents[1]
        source_paths = (
            Path(__file__),
            Path(__file__).with_name("core.py"),
            Path(__file__).with_name("lattice.py"),
            Path(__file__).with_name("shields.py"),
            Path(__file__).with_name("crystallize_injection.py"),
        )
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            dirty = bool(subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain",
                 "--untracked-files=no"],
                check=True, capture_output=True, text=True,
            ).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        return {
            "commit": commit,
            "dirty": dirty,
            "source_sha256": hashes,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "seed": self.config.seed,
            "seed_streams": {
                "simulation": "numpy.default_rng(seed); shared by initialization and lattice dynamics",
            },
        }

    def step(self) -> dict:
        """Execute one transactional global update, with adaptive repair."""
        if self.circuit_breaker:
            return {
                "status": "circuit_breaker_open",
                "step": self.current_step,
            }

        observation = self.observer.observe(
            self.lattice.state, self.lattice.reference, self.current_step
        )

        drive_time = (
            self.accepted_time
            if self.config.drive_time_mode == "accepted_time"
            else self.current_step * self.config.lattice.dt
        )
        action = self.calibrator.propose(
            observation,
            self.current_step,
            self.config.schumann,
            self.config.schumann_hz,
            self.config.schumann_amplitude,
            self.config.lattice.dt,
            drive_time_seconds=drive_time,
        )

        # The calibrator never changes these; log the values in effect so the
        # audit trail does not show misleading zeros.
        lattice_cfg = self.config.lattice
        action.coupling = lattice_cfg.coupling
        action.dephasing = lattice_cfg.dephasing
        action.mirror_strength = lattice_cfg.mirror_strength

        effective = {
            "dt": lattice_cfg.dt,
            "mirror_strength": lattice_cfg.mirror_strength,
            "coupling": lattice_cfg.coupling,
            "dephasing": lattice_cfg.dephasing,
        }
        candidate = self.lattice.propose_step(action.total_signal)
        accepted, reason, report = self._validate_with_telemetry(
            candidate, effective
        )
        attempts = [{
            "step": self.current_step + 1,
            "attempt": 0,
            "proposal": "initial",
            "dt": effective["dt"],
            "mirror_strength": effective["mirror_strength"],
            "coupling": effective["coupling"],
            "dephasing": effective["dephasing"],
            "drive_time_seconds": drive_time,
            "drive_time_mode": self.config.drive_time_mode,
            "intervention_type": "none",
            "accepted": bool(accepted),
            "reason": reason,
            **report,
        }]
        repair_attempts: list[dict] = []
        accepted_via_repair = False

        # Pairwise-guard reference: a retry variant is judged against the
        # FULL-step candidate's minimum divergence, not its own. A
        # smaller-dt retry can move a collided pair apart (pair fidelity
        # 0.95 -> 0.9457 at dt x0.5 vs -> 0.9525 at full dt), so judging a
        # variant by its own divergence would launder a collided step
        # past the guard. If the full step is collided, no variant can
        # rescue the step. (Skipped only when the full-step candidate is
        # an invalid density matrix, where the reference divergence is
        # meaningless and the rejection is a separate failure mode.)
        # The reference uses the FULL pairwise condition (threshold AND
        # margin): a candidate whose min divergence is low only because
        # the whole lattice is synchronising is not a collision.
        # Metrics may be empty for an invalid candidate; default to 1.0
        # (not collided) so the reference is well-defined there.
        pairwise_threshold = self.config.guards.min_pairwise_divergence
        _metrics = report.get("metrics") or {}
        full_min_divergence = _metrics.get("min_divergence", 1.0)
        full_mean_divergence = _metrics.get("mean_divergence", 1.0)
        pairwise_reference_ok = (
            pairwise_threshold is None
            or reason.startswith("invalid_density_matrix")
            or not self.observer.pairwise_collapsed_values(
                full_min_divergence, full_mean_divergence
            )
        )
        full_report = report

        if not accepted and self.config.adaptive_retry:
            for k in range(self.config.max_retries):
                dt_eff = lattice_cfg.dt * self.config.retry_dt_factors[k]
                if self.config.retry_policy == "dt_only":
                    mirror_eff = lattice_cfg.mirror_strength
                    coupling_eff = lattice_cfg.coupling
                else:
                    mirror_eff = lattice_cfg.mirror_strength * self.config.retry_mirror_factors[k]
                    coupling_eff = lattice_cfg.coupling * self.config.retry_coupling_factors[k]
                effective = {
                    "dt": dt_eff,
                    "mirror_strength": mirror_eff,
                    "coupling": coupling_eff,
                    "dephasing": lattice_cfg.dephasing,
                }
                variant = self.lattice.propose_step(
                    action.total_signal,
                    dt=dt_eff,
                    mirror_strength=mirror_eff,
                    coupling=coupling_eff,
                )
                ok, variant_reason, variant_report = self._validate_with_telemetry(
                    variant, effective
                )
                self.repair_attempts_total += 1
                intervention = (
                    "numerical_backtracking"
                    if mirror_eff == lattice_cfg.mirror_strength
                    and coupling_eff == lattice_cfg.coupling
                    else "model_intervention"
                )
                # The variant may pass on its own trajectory, but if the
                # full-step reference is collided the step is a collision
                # and no smaller-dt variant may launder it past the guard.
                # The reason stays the plain guard name (the attempt and
                # step records already carry the full-step reference in
                # pairwise_reference_ok / pairwise_reference_min_divergence).
                laundered = ok and not pairwise_reference_ok
                if laundered:
                    ok = False
                    variant_reason = "pairwise_divergence_below_minimum"
                attempt = {
                    "step": self.current_step + 1,
                    "attempt": k + 1,
                    "proposal": "repair",
                    **effective,
                    "drive_time_seconds": drive_time,
                    "drive_time_mode": self.config.drive_time_mode,
                    "intervention_type": intervention,
                    "accepted": bool(ok),
                    "reason": variant_reason,
                    **variant_report,
                    "pairwise_reference_ok": pairwise_reference_ok,
                    "pairwise_reference_min_divergence": full_min_divergence,
                }
                attempts.append(attempt)
                repair_attempts.append(attempt)
                # A laundered variant is rejected on the full-step
                # reference, so the rejection record shows the collision.
                report = full_report if laundered else variant_report
                if ok:
                    candidate, accepted, reason = variant, True, "ok"
                    accepted_via_repair = True
                    self.repair_accepts += 1
                    # Calibration log keeps the stock values in effect;
                    # the variant's dt/mirror/coupling are in
                    # repair_attempts, not in the calibrator's proposal.
                    break
                reason = variant_reason

        final_step_reason = "ok" if accepted else reason
        self.attempt_ledger.extend(attempts)

        if accepted:
            self.lattice.commit(candidate)
            self.accepted_time += effective["dt"]
            self.failures = 0
        else:
            # One failure per step, counted only after all repair
            # attempts failed — never one per identical retry.
            self.failures += 1
            self.rejections.append(
                {
                    "step": self.current_step + 1,
                    "reason": final_step_reason,
                    "final_step_reason": final_step_reason,
                    "failed_guard": report["failed_guard"],
                    "offending_cells": report["offending_cells"],
                    "offending_by_guard": report["offending_by_guard"],
                    "metrics": report["metrics"],
                    "thresholds": report["thresholds"],
                    "repair_attempts": repair_attempts,
                    "attempts": attempts,
                }
            )
            if self.failures >= self.config.failure_limit:
                self.circuit_breaker = True

        self.calibrator.mark(action, accepted, final_step_reason)
        self.calibrations.append(action.to_dict())

        post = self.observer.observe(
            self.lattice.state, self.lattice.reference, self.current_step + 1
        )
        self.observations.append(post.to_dict())
        self.current_step += 1

        return {
            "status": "accepted" if accepted else "rejected",
            "reason": final_step_reason,
            "final_step_reason": final_step_reason,
            "step": self.current_step,
            "observation": post.to_dict(),
            "calibration": action.to_dict(),
            "accepted_via_repair": accepted_via_repair,
            "repair_attempts": repair_attempts,
            "attempts": attempts,
            "accepted_time": self.accepted_time,
            "pairwise_reference_ok": pairwise_reference_ok,
            "pairwise_reference_min_divergence": full_min_divergence,
        }

    def run(self) -> dict:
        """Run the configured number of further steps, stopping if the breaker opens.

        The step counter and the logs continue from where the last call
        stopped; a second call does not restart them.

        Before the first step, the pairwise guard (when enabled) is also
        checked against the INITIAL state: a run that starts collided is
        caught at step 0 — the breaker latches with 0 accepted steps —
        instead of being discovered only after a step.
        """
        if self.config.steps > 0 and not self.circuit_breaker:
            threshold = self.config.guards.min_pairwise_divergence
            if threshold is not None:
                initial = self.observer.observe(
                    self.lattice.state, self.lattice.reference, 0
                )
                if self.observer.pairwise_collapsed(initial):
                    report = self.observer.guard_report(initial)
                    self.rejections.append(
                        {
                            "step": 0,
                            "reason": "pairwise_divergence_below_minimum",
                            "failed_guard": "pairwise_divergence_below_minimum",
                            "offending_cells": report["offending_cells"],
                            "offending_by_guard": report["offending_by_guard"],
                            "metrics": report["metrics"],
                            "thresholds": report["thresholds"],
                            "repair_attempts": [],
                        }
                    )
                    self.failures = self.config.failure_limit
                    self.circuit_breaker = True
                    return self.summary()
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
            "accepted_time": self.accepted_time,
            "accepted_steps": accepted,
            "rejected_steps": len(self.calibrations) - accepted,
            "guards": self.config.guards_dict(),
            "operator_mode": self.config.lattice.operator_mode,
            "adaptive_retry": self.config.adaptive_retry,
            "rejections": self.rejections,
            "repair_attempts_total": self.repair_attempts_total,
            "repair_accepts": self.repair_accepts,
            "attempt_ledger": self.attempt_ledger,
            "configuration": asdict(self.config),
            "provenance": self._provenance(),
            "final": final,
            "observations": self.observations,
            "calibrations": self.calibrations,
        }


def run_simulation(config: EngineConfig | None = None) -> dict:
    """Convenience function for running one complete simulation."""
    return MembraneArena3x3(config).run()
