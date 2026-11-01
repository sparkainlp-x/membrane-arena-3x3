import json
from pathlib import Path

import pytest

from scripts.scientific_verification import run

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
        )


def test_full_profile_cannot_run_all_splits_at_once(tmp_path):
    with pytest.raises(ValueError, match="must be run by split"):
        run(
            ROOT / "verification_configs" / "full.json",
            tmp_path,
            "all",
            "reproduction command",
            None,
            None,
        )
