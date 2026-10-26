"""docs/RESULTS-injection.md is generated from the measured files (SYNTHETIC)."""

import os
import re

from experiments import build_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_committed_results_match_the_generated_output():
    with open(os.path.join(ROOT, "docs", "RESULTS-injection.md"), encoding="utf-8") as fh:
        committed = fh.read()
    assert committed == build_results.render(), "run: python -m experiments.build_results --write"


def test_results_report_every_hypothesis_and_the_honest_limits():
    text = build_results.render()
    for i in range(1, 13):
        assert f"| C{i} |" in text
    for needle in ("Honest limits", "Deviation 1", "does not cure the breaker", "two-forecast (primary)"):
        assert needle in text
    assert "pending" not in text.lower()


def test_reproduce_script_commands_reference_existing_files_and_makefile_calls_it():
    import importlib.util

    spec = importlib.util.spec_from_file_location("reproduce_results", os.path.join(ROOT, "scripts", "reproduce_results.py"))
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    cmds = rr.plan(offline=False)
    assert rr.plan(offline=True) == rr.OFFLINE and cmds[-1] == ["-m", "experiments.build_results", "--write"]
    for cmd in cmds:
        if cmd[0] == "-m":
            assert os.path.exists(os.path.join(ROOT, *cmd[1].split(".")) + ".py"), cmd
        else:
            assert os.path.exists(os.path.join(ROOT, cmd[0])), cmd
    with open(os.path.join(ROOT, "Makefile"), encoding="utf-8") as fh:
        make = fh.read()
    assert "scripts/reproduce_results.py --offline" in make and "results:" in make


def test_citation_zenodo_and_package_versions_agree():
    def grab(path, pattern):
        with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
            return re.search(pattern, fh.read(), re.M).group(1)

    v = grab("pyproject.toml", r'^version = "([^"]+)"')
    assert grab("CITATION.cff", r"^version: (\S+)") == v
    assert grab(".zenodo.json", r'"version": "([^"]+)"') == v
