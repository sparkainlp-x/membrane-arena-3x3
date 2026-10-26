"""Regenerate every measured result behind docs/RESULTS-injection.md (SYNTHETIC).

Usage (from the repository root):
    python scripts/reproduce_results.py            # re-collect all raw runs, re-run every analysis, rebuild the file
    python scripts/reproduce_results.py --offline  # re-run the analyses from the cached raw runs, rebuild the file
    python scripts/reproduce_results.py --list     # print the commands without running them

``make results`` and ``make results-offline`` call this script. OMP_NUM_THREADS=1
is set for every step. The full run takes tens of minutes on one core
(re-collecting 910 runs, the 50-seed sweep and the note reconciliation); the
offline run takes seconds.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLLECT = [["-m", "experiments.injection_runs", "--fresh"]]
HEAVY = [
    ["scripts/injection_sweep.py", "--write", "docs/injection_results.md"],
    ["scripts/injection_multiseed.py", "--seeds", "50", "--write"],
    ["-m", "experiments.reconcile_note", "--write"],
]
OFFLINE = [
    ["-m", "experiments.run_prereg_analysis", "--write"],
    ["-m", "experiments.run_event_structure", "--write"],
    ["-m", "experiments.compare_alignment", "--write"],
    ["-m", "experiments.run_controls", "--write"],
    ["scripts/injection_multiseed.py", "--render-only", "--write"],
    ["-m", "experiments.build_results", "--write"],
]


def plan(offline: bool) -> list[list[str]]:
    return OFFLINE if offline else COLLECT + HEAVY + OFFLINE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for cmd in plan(args.offline):
        print("+ python " + " ".join(cmd), flush=True)
        if not args.list:
            subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
