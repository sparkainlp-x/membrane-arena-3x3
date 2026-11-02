"""Reproduce the opt-in scientific-verification smoke/full protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from membrane_arena_3x3.engine import EngineConfig, MembraneArena3x3
from membrane_arena_3x3.lattice import LatticeConfig
from membrane_arena_3x3.shields import GuardConfig
from membrane_arena_3x3.wave_memory import run_wave_memory


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_provenance() -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain",
             "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "dirty": dirty}


def _source_hashes() -> dict[str, str]:
    sources = (
        ROOT / "membrane_arena_3x3" / "engine.py",
        ROOT / "membrane_arena_3x3" / "lattice.py",
        ROOT / "membrane_arena_3x3" / "shields.py",
        ROOT / "membrane_arena_3x3" / "core.py",
        ROOT / "membrane_arena_3x3" / "crystallize_injection.py",
        ROOT / "membrane_arena_3x3" / "wave_memory.py",
        Path(__file__),
    )
    return {
        path.relative_to(ROOT).as_posix(): _sha256(path.read_bytes())
        for path in sources
    }


def _source_bundle_sha256(source_hashes: dict[str, str]) -> str:
    return _sha256(_json_bytes(source_hashes))


def _collect_engine(seed: int, config: dict) -> tuple[list[dict], dict[str, dict]]:
    common = config["engine"]
    arms = common["ablation_arms"]
    trajectories: dict[str, dict] = {}
    rows = []
    for arm, overrides in arms.items():
        lattice = {**common["lattice"], **overrides}
        engine_config = EngineConfig(
            lattice=LatticeConfig(**lattice),
            guards=GuardConfig(**common["guards"]),
            steps=int(common["max_attempts"]),
            seed=seed,
            failure_limit=int(common["failure_limit"]),
            adaptive_retry=bool(common["adaptive_retry"]),
            retry_policy=common["retry_policy"],
            drive_time_mode=common["drive_time_mode"],
        )
        arena = MembraneArena3x3(engine_config)
        states = {0.0: arena.lattice.state.copy()}
        proposals = 0
        for _ in range(int(common["max_attempts"])):
            step_result = arena.step()
            if step_result["status"] == "circuit_breaker_open":
                break
            proposals += len(step_result["attempts"])
            if step_result["status"] == "accepted":
                states[round(arena.accepted_time, 12)] = arena.lattice.state.copy()
            if arena.accepted_time >= float(common["target_accepted_time"]):
                break
        summary = arena.summary()
        causes: dict[str, int] = {}
        for attempt in summary["attempt_ledger"]:
            for cause in attempt["guard_violations"]:
                causes[cause] = causes.get(cause, 0) + 1
        trajectories[arm] = states
        rows.append({
            "experiment": "engine_ablation",
            "seed": seed,
            "arm": arm,
            "operator_mode": lattice["operator_mode"],
            "steps_executed": summary["steps_executed"],
            "accepted_steps": summary["accepted_steps"],
            "rejected_steps": summary["rejected_steps"],
            "accepted_time": summary["accepted_time"],
            "target_accepted_time": common["target_accepted_time"],
            "target_reached": summary["accepted_time"] >= common["target_accepted_time"],
            "attempts": len(summary["attempt_ledger"]),
            "repair_attempts": summary["repair_attempts_total"],
            "repair_accepts": summary["repair_accepts"],
            "candidate_proposals": proposals,
            "estimated_dense_matrix_element_operations": (
                proposals * 9 * lattice["dim"] ** 3
            ),
            "guard_causes": causes,
            "final_reason": (
                summary["attempt_ledger"][-1]["reason"]
                if summary["attempt_ledger"] else "not_run"
            ),
        })

    reference = trajectories["reference"]
    for arm, states in trajectories.items():
        common_times = sorted(set(reference).intersection(states))
        for elapsed in common_times:
            if elapsed == 0.0:
                continue
            delta = states[elapsed] - reference[elapsed]
            rows.append({
                "experiment": "equal_accepted_time_trajectory",
                "seed": seed,
                "arm": arm,
                "common_accepted_time": elapsed,
                "trajectory_error_frobenius_per_cell": float(
                    np.linalg.norm(delta) / np.sqrt(9)
                ),
                "reference_and_arm_share_seeded_initial_state": True,
            })
    return rows, trajectories


def _percentile_interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([777, len(values)]))
    samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def _summarize(rows: list[dict], seeds: list[int], config: dict) -> dict:
    wave_rows = [row for row in rows if row["experiment"] == "wave_memory"]
    engine_rows = [row for row in rows if row["experiment"] == "engine_ablation"]
    trajectory_rows = [
        row for row in rows
        if row["experiment"] == "equal_accepted_time_trajectory"
    ]
    intervention_names = sorted({row["intervention"] for row in wave_rows})
    wave_summary = {}
    for intervention in intervention_names:
        selected = [r for r in wave_rows if r["intervention"] == intervention]
        wave_summary[intervention] = {
            "trials": len(selected),
            "decoding_errors": sum(bool(r["decoding_error"]) for r in selected),
            "decoding_error_rate": (
                float(np.mean([r["decoding_error"] for r in selected]))
                if selected else None
            ),
            "input_energy_mean": float(np.mean([r["input_energy"] for r in selected])),
            "condition_number_mean": float(
                np.mean([r["condition_number"] for r in selected])
            ),
        }
    strata = []
    stratum_keys = sorted({
        (row["intervention"], row["delay"], row["crosstalk"])
        for row in wave_rows
    })
    for intervention, delay, cross in stratum_keys:
        selected = [
            row for row in wave_rows
            if row["intervention"] == intervention
            and row["delay"] == delay
            and row["crosstalk"] == cross
        ]
        strata.append({
            "intervention": intervention,
            "delay": delay,
            "crosstalk": cross,
            "trials": len(selected),
            "decoding_error_rate": float(
                np.mean([row["decoding_error"] for row in selected])
            ),
            "condition_number_mean": float(
                np.mean([row["condition_number"] for row in selected])
            ),
        })

    paired_effects = {}
    for intervention in intervention_names:
        if intervention == "no_intervention":
            continue
        differences = []
        for seed in seeds:
            base = [r["decoding_error"] for r in wave_rows
                    if r["seed"] == seed and r["intervention"] == "no_intervention"]
            changed = [r["decoding_error"] for r in wave_rows
                       if r["seed"] == seed and r["intervention"] == intervention]
            if base and len(base) == len(changed):
                differences.append(float(np.mean(changed) - np.mean(base)))
        paired_effects[intervention] = {
            "experimental_unit": "seed-level paired mean decoding-error difference",
            "n_units": len(differences),
            "mean_difference": float(np.mean(differences)) if differences else None,
            "bootstrap_95_percentile_interval": _percentile_interval(differences),
            "p_value_or_independence_claim": None,
        }

    trajectory_effects = {}
    for arm in sorted({r["arm"] for r in engine_rows}):
        errors = [r["trajectory_error_frobenius_per_cell"] for r in trajectory_rows
                  if r["arm"] == arm]
        trajectory_effects[arm] = {
            "matched_seed_timepoints": len(errors),
            "mean_trajectory_error": float(np.mean(errors)) if errors else None,
            "max_trajectory_error": float(np.max(errors)) if errors else None,
        }
    engine_summary = {}
    for arm in sorted({row["arm"] for row in engine_rows}):
        selected = [row for row in engine_rows if row["arm"] == arm]
        causes: dict[str, int] = {}
        for row in selected:
            for name, count in row["guard_causes"].items():
                causes[name] = causes.get(name, 0) + count
        engine_summary[arm] = {
            "runs": len(selected),
            "target_reached": sum(bool(row["target_reached"]) for row in selected),
            "mean_achieved_accepted_time": float(
                np.mean([row["accepted_time"] for row in selected])
            ),
            "mean_attempts": float(np.mean([row["attempts"] for row in selected])),
            "mean_candidate_proposals": float(
                np.mean([row["candidate_proposals"] for row in selected])
            ),
            "mean_estimated_dense_matrix_element_operations": float(
                np.mean([
                    row["estimated_dense_matrix_element_operations"]
                    for row in selected
                ])
            ),
            "guard_causes": causes,
        }
    return {
        "profile": config["profile"],
        "seed_split": config["active_split"],
        "seed_count": len(seeds),
        "engine_ablation_runs": len(engine_rows),
        "equal_accepted_time_comparisons": len(trajectory_rows),
        "wave_memory_rows": len(wave_rows),
        "wave_memory": wave_summary,
        "wave_memory_strata": strata,
        "engine_ablation": engine_summary,
        "paired_effects": paired_effects,
        "trajectory_effects": trajectory_effects,
        "experimental_unit_note": (
            "Seed-level pairing is descriptive; the seed count alone does not "
            "establish independence or statistical significance."
        ),
        "held_out_executed": config["active_split"] == "held_out",
        "held_out_tuning": "none",
        "threshold_sensitivity": (
            "Not tuned in this benchmark; no guard thresholds were changed."
        ),
    }


def _run_scope_note(summary: dict) -> str:
    if summary["profile"] == "smoke":
        return (
            "This is a smoke artifact over development seeds only; full "
            "development, validation, held-out, and physical-validation runs "
            "are not part of this artifact."
        )
    return (
        f"This full-profile artifact contains the `{summary['seed_split']}` split "
        "only. Other seed splits require separate runs; this report does not "
        "assert that they have been run. Physical validation was not run."
    )


def _report(summary: dict, command: str, hashes: dict) -> str:
    lines = [
        "# Scientific verification smoke/full run",
        "",
        "All results are synthetic classical software measurements. This is not "
        "an NV hardware simulator, physical memory model, or quantum-channel claim.",
        "",
        f"- Profile/split: `{summary['profile']}` / `{summary['seed_split']}`",
        f"- Seeds: {summary['seed_count']}",
        f"- Engine ablation runs: {summary['engine_ablation_runs']}",
        f"- Equal accepted-time trajectory comparisons: "
        f"{summary['equal_accepted_time_comparisons']}",
        f"- Wave-memory rows: {summary['wave_memory_rows']}",
        f"- Held-out executed: `{summary['held_out_executed']}`; tuning: `none`",
        "",
        "## Wave-memory outcomes",
        "",
        "| intervention | trials | decoding errors | error rate |",
        "|---|---:|---:|---:|",
    ]
    for arm, result in summary["wave_memory"].items():
        lines.append(
            f"| {arm} | {result['trials']} | {result['decoding_errors']} | "
            f"{result['decoding_error_rate']:.6f} |"
        )
    lines.extend([
        "",
        "### Delay/crosstalk strata",
        "",
        "| intervention | delay | crosstalk | trials | error rate | mean condition number |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for result in summary["wave_memory_strata"]:
        lines.append(
            f"| {result['intervention']} | {result['delay']} | "
            f"{result['crosstalk']:.3f} | {result['trials']} | "
            f"{result['decoding_error_rate']:.6f} | "
            f"{result['condition_number_mean']:.6f} |"
        )
    lines.extend([
        "",
        "The signed integer alphabet has canonical zero, and nonzero benchmark "
        "inputs have matched energy before the dissipative-reset reference. "
        "Noise variance is normalized so expected total noise power equals "
        "input energy divided by `10^(SNR_dB/10)`. Channel drift, crosstalk, "
        "conditioning, delay, and errors are reported in `raw_results.csv`.",
        "",
        "Detector-side inverse cancellation is an oracle benchmark intervention; "
        "it changes only the decoded observation, not the stored input. The "
        "coherent arm applies a unitary and its inverse. The dissipative reset "
        "replaces every input by zero and is expected to erase nonzero messages; "
        "it does not model a unitary reset pulse.",
        "",
        "## Engine outcomes and limitations",
        "",
        "### Engine ablation outcomes",
        "",
        "| arm | target reached / runs | mean accepted time | mean attempts | estimated matrix operations | guard causes |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for arm, result in summary["engine_ablation"].items():
        lines.append(
            f"| {arm} | {result['target_reached']} / {result['runs']} | "
            f"{result['mean_achieved_accepted_time']:.6f} | "
            f"{result['mean_attempts']:.2f} | "
            f"{result['mean_estimated_dense_matrix_element_operations']:.0f} | "
            f"`{json.dumps(result['guard_causes'], sort_keys=True)}` |"
        )
    lines.extend([
        "",
        "### Equal-time trajectory differences",
        "",
        "| arm | matched timepoints | mean error | maximum error |",
        "|---|---:|---:|---:|",
    ])
    for arm, result in summary["trajectory_effects"].items():
        mean = result["mean_trajectory_error"]
        maximum = result["max_trajectory_error"]
        lines.append(
            f"| {arm} | {result['matched_seed_timepoints']} | "
            f"{'not available' if mean is None else f'{mean:.8g}'} | "
            f"{'not available' if maximum is None else f'{maximum:.8g}'} |"
        )
    lines.extend([
        "",
        "`raw_results.csv` reports accepted time, proposal/repair counts, guard "
        "causes, a deterministic matrix-operation cost proxy, and trajectory "
        "error only at matched accepted times. Missing common times are not "
        "interpolated. The operation count is a proxy, not measured wall time.",
        "",
        summary["experimental_unit_note"],
        "Paired effects and percentile intervals are descriptive at the "
        "seed-level experimental unit; no significance or independence claim "
        "is made.",
        (
            "Smoke outcomes are development-only and not confirmatory."
            if summary["profile"] == "smoke"
            else "Full-profile outcomes are split-specific and not confirmatory."
        ),
        _run_scope_note(summary),
        "",
        "The stock retry policy remains the historical model-changing policy. "
        "A `dt_only` option is available for numerical backtracking; mirror and "
        "coupling changes are model interventions, not accuracy improvements. "
        "The default drive retains legacy `step_index * nominal_dt` phase; "
        "`accepted_time` is an explicit alternate.",
        "",
        "PR #10 remains an open dependency. No mirror-reference implementation "
        "was copied or wired into this branch. Its proposed map is an exact "
        "mirror-only map, not an exact combined solver or physical quantum "
        "channel. Other operators still require timestep-convergence analysis.",
        "",
        "Historical reproduction uses `retry_policy=legacy_model_changing`, "
        "`drive_time_mode=legacy_step_index`, fixed per-step mirror strength, "
        "the stock `operator_mode=legacy`, and the default seed stream "
        "`numpy.default_rng(seed)`. These settings are recorded in "
        "`configuration.json`; changing them does not reproduce historical runs.",
        "",
        "## Reproduction and hashes",
        "",
        f"```bash\n{command}\n```",
        "",
        f"- Configuration SHA-256: `{hashes['configuration']}`",
        f"- Protocol SHA-256: `{hashes['protocol']}`",
        f"- Source bundle SHA-256: `{hashes['source_bundle']}`",
        "",
        "Runtime and timestamps are intentionally absent; deterministic numerical "
        "comparisons exclude runtime/environment metadata and use the exact raw "
        "seed/configuration. See `manifest.json` and `checksums.sha256`.",
        "",
        "## Scope / unresolved validation",
        "",
        "This run does not establish controller effectiveness beyond the stated "
        "toy outcomes, threshold robustness, physical validity, atomic addressing, "
        "zero-power memory, thousands of independent modes, gravity, biology, or "
        "consciousness. Numerical validity failures remain fail-closed. The "
        "mirror integration and hardware validation remain unrun.",
    ])
    return "\n".join(lines) + "\n"


def run(config_path: Path, out_dir: Path, split: str, command: str,
        expected_config_hash: str | None, expected_protocol_hash: str | None,
        expected_source_hash: str | None) -> None:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    config_hash = _sha256(config_bytes)
    protocol_path = ROOT / "docs" / "SCIENTIFIC-VERIFICATION.md"
    protocol_hash = _sha256(protocol_path.read_bytes())
    source_hashes = _source_hashes()
    source_bundle_hash = _source_bundle_sha256(source_hashes)
    seed_splits = config["seed_splits"]
    profile = config["profile"]

    if profile == "full" and split == "all":
        raise ValueError("full profile must be run by split; held-out requires frozen hashes")
    if split not in ("all", "development", "validation", "held_out"):
        raise ValueError("split must be all, development, validation, or held_out")
    if split == "held_out":
        if not expected_config_hash or not expected_protocol_hash or not expected_source_hash:
            raise ValueError("held-out execution requires frozen config, protocol, and source SHA-256 values")
        if (
            expected_config_hash != config_hash
            or expected_protocol_hash != protocol_hash
            or expected_source_hash != source_bundle_hash
        ):
            raise ValueError(
                "held-out configuration/protocol/source hash does not match frozen values"
            )
    elif expected_config_hash or expected_protocol_hash or expected_source_hash:
        raise ValueError("frozen hash arguments are only valid for held-out execution")

    seeds = (
        seed_splits["development"]
        + seed_splits["validation"]
        + seed_splits["held_out"]
        if split == "all" else seed_splits[split]
    )
    if not seeds:
        raise ValueError(f"the {split} split has no configured seeds")
    config["active_split"] = split
    rows = []
    for seed in seeds:
        engine_rows, _ = _collect_engine(int(seed), config)
        rows.extend(engine_rows)
        rows.extend(run_wave_memory(int(seed), config["wave_memory"]))

    summary = _summarize(rows, [int(seed) for seed in seeds], config)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path_out = out_dir / "configuration.json"
    config_path_out.write_bytes(config_bytes)
    raw_path = out_dir / "raw_results.csv"
    columns = sorted({key for row in rows for key in row})
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })

    summary_path = out_dir / "summary.json"
    summary_path.write_bytes(_json_bytes(summary))
    manifest = {
        "command": command,
        "commit_and_dirty_state": _git_provenance(),
        "profile": profile,
        "split": split,
        "configuration_sha256": config_hash,
        "protocol_sha256": protocol_hash,
        "source_sha256": source_hashes,
        "source_bundle_sha256": source_bundle_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "seed_splits": seed_splits,
        "seed_streams": {
            "engine": "numpy.default_rng(seed), shared by initialization and lattice dynamics",
            "wave_memory": {
                "message": "SeedSequence([seed, 101])",
                "channel_drift": "SeedSequence([seed, trial, crosstalk_index, delay_index, 201])",
                "noise": "SeedSequence([seed, trial, crosstalk_index, delay_index, 301])",
                "codebook": "SeedSequence([seed, 401])",
            },
        },
        "held_out_freeze_check": (
            {"configuration_sha256": expected_config_hash,
             "protocol_sha256": expected_protocol_hash,
             "source_bundle_sha256": expected_source_hash}
            if split == "held_out" else None
        ),
        "runtime_fields_in_deterministic_outputs": False,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    hashes = {
        "configuration": config_hash,
        "protocol": protocol_hash,
        "source_bundle": source_bundle_hash,
    }
    report_path = out_dir / "report.md"
    report_path.write_text(_report(summary, command, hashes), encoding="utf-8")
    files = (
        config_path_out, manifest_path, raw_path, summary_path, report_path,
    )
    checksums = "".join(
        f"{_sha256(path.read_bytes())}  {path.name}\n" for path in files
    )
    (out_dir / "checksums.sha256").write_text(checksums, encoding="utf-8")


def _reproduction_command(
    config: str,
    out_dir: str,
    split: str,
    expected_config_hash: str | None = None,
    expected_protocol_hash: str | None = None,
    expected_source_hash: str | None = None,
) -> str:
    command_args = [
        sys.executable,
        "scripts/scientific_verification.py",
        "--config", config,
        "--out-dir", out_dir,
        "--split", split,
    ]
    for option, value in (
        ("--expected-config-sha256", expected_config_hash),
        ("--expected-protocol-sha256", expected_protocol_hash),
        ("--expected-source-sha256", expected_source_hash),
    ):
        if value:
            command_args.extend([option, value])
    return shlex.join(command_args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=("all", "development", "validation", "held_out"),
                        default="all")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    command = _reproduction_command(
        args.config,
        args.out_dir,
        args.split,
        args.expected_config_sha256,
        args.expected_protocol_sha256,
        args.expected_source_sha256,
    )
    run(
        Path(args.config), Path(args.out_dir), args.split, command,
        args.expected_config_sha256, args.expected_protocol_sha256,
        args.expected_source_sha256,
    )
    print(f"wrote verification artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
