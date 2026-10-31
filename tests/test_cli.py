"""Command-line entry point."""

import json

import pytest

from membrane_arena_3x3.cli import main

SUMMARY_KEYS = {
    "status",
    "steps_executed",
    "accepted_steps",
    "rejected_steps",
    "final_mean_purity",
    "final_max_divergence",
    "final_max_residual",
    "final_min_purity",
    "final_stability_penalty",
    "final_mean_off_diagonal_coherence",
    "guards",
    "operator_mode",
    "repair_accepts",
}


def run_cli(monkeypatch, capsys, *argv):
    monkeypatch.setattr("sys.argv", ["membrane-arena-3x3", *argv])
    main()
    return json.loads(capsys.readouterr().out)


def test_cli_prints_the_summary_and_writes_json(monkeypatch, capsys, tmp_path):
    out = tmp_path / "r.json"
    summary = run_cli(monkeypatch, capsys, "--steps", "3", "--json", str(out))
    assert set(summary) == SUMMARY_KEYS
    assert summary["steps_executed"] == 3
    full = json.loads(out.read_text(encoding="utf-8"))
    assert len(full["observations"]) == 3


def test_cli_zero_steps_does_not_crash(monkeypatch, capsys):
    summary = run_cli(monkeypatch, capsys, "--steps", "0")
    assert summary["status"] == "no_steps"
    assert summary["final_min_purity"] is None


def test_cli_reports_an_aliasing_setting_as_a_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["membrane-arena-3x3", "--schumann", "--dt", "0.1"]
    )
    with pytest.raises(SystemExit) as info:
        main()
    assert info.value.code == 2
    assert "Nyquist" in capsys.readouterr().err
