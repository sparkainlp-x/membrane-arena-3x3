import json
import hashlib
from pathlib import Path

import pytest

from scripts.scientific_verification import (
    _reproduction_command,
    _run_scope_note,
    _source_bundle_sha256,
    _source_hashes,
    run,
)

ROOT = Path(__file__).resolve().parents[1]


def test_full_seed_splits_are_disjoint_and_complete():
    config = json.loads((ROOT / "verification_configs" / "full.json").read_text())
    splits = config["seed_splits"]
    assert splits["development"] == list(range(20))
    assert splits["validation"] == list(range(20, 50))
    assert splits["held_out"] == list(range(50, 100))
    assert len(set(splits["development"] + splits["validation"] + splits["held_out"])) == 100


def test_held_out_requires_matching_frozen_hashes(tmp_path):
    with pytest.raises(ValueError, match="hash does not match"):
        run(
            ROOT / "verification_configs" / "full.json",
            tmp_path,
            "held_out",
            "reproduction command",
            "not-the-config-hash",
            "not-the-protocol-hash",
            "not-the-source-hash",
        )


def test_held_out_requires_and_checks_frozen_source_hash(tmp_path):
    config_path = ROOT / "verification_configs" / "full.json"
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    protocol_hash = hashlib.sha256(
        (ROOT / "docs" / "SCIENTIFIC-VERIFICATION.md").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="requires frozen"):
        run(
            config_path,
            tmp_path,
            "held_out",
            "reproduction command",
            config_hash,
            protocol_hash,
            None,
        )
    with pytest.raises(ValueError, match="source hash"):
        run(
            config_path,
            tmp_path,
            "held_out",
            "reproduction command",
            config_hash,
            protocol_hash,
            "not-the-source-hash",
        )
    assert _source_bundle_sha256(_source_hashes())


def test_full_profile_cannot_run_all_splits_at_once(tmp_path):
    with pytest.raises(ValueError, match="must be run by split"):
        run(
            ROOT / "verification_configs" / "full.json",
            tmp_path,
            "all",
            "reproduction command",
            None,
            None,
            None,
        )


def test_report_scope_depends_on_actual_profile_and_split():
    assert "smoke artifact" in _run_scope_note({
        "profile": "smoke",
        "seed_split": "all",
    })
    full_note = _run_scope_note({
        "profile": "full",
        "seed_split": "held_out",
    })
    assert "full-profile artifact contains the `held_out` split only" in full_note


def test_held_out_reproduction_command_contains_all_freeze_hashes():
    command = _reproduction_command(
        "verification_configs/full.json",
        "verification_results/held-out",
        "held_out",
        "config-hash",
        "protocol-hash",
        "source-hash",
    )
    for option, value in (
        ("--expected-config-sha256", "config-hash"),
        ("--expected-protocol-sha256", "protocol-hash"),
        ("--expected-source-sha256", "source-hash"),
    ):
        assert option in command and value in command
