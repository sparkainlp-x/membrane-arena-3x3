"""Exploratory: how injection events are laid out inside a run (SYNTHETIC).

Not pre-registered. Every run ends when 5 consecutive steps are rejected, so
the last events of a run are failures by construction. This script separates
that terminal streak from the rest and shows the pass rate by event index.
Usage: python -m experiments.run_event_structure [--write]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from experiments.injection_runs import RESULTS_DIR, collect_set
from experiments.regimes import REGIMES


def terminal_streak(events) -> int:
    k = 0
    for _, passed in reversed(events):
        if passed:
            break
        k += 1
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    runs = collect_set("confirm")
    L = ["# Event structure inside runs (SYNTHETIC, exploratory, confirm seeds; experiments/run_event_structure.py)", "",
         "| regime | mean events per run | mean terminal failure streak | pass rate, all events | pass rate, events before the terminal streak |",
         "|---|---|---|---|---|"]
    idx_rows = []
    for reg in REGIMES:
        rr = [r for r in runs if r["regime"] == reg]
        ev = [r["events"] for r in rr]
        streaks = [terminal_streak(e) for e in ev]
        tot = sum(len(e) for e in ev)
        ps = sum(sum(1 for _, p in e if p) for e in ev)
        body = [e[: len(e) - s] for e, s in zip(ev, streaks)]
        btot = sum(len(b) for b in body)
        bps = sum(sum(1 for _, p in b if p) for b in body)
        L.append(f"| {reg} | {tot/len(rr):.1f} | {np.mean(streaks):.2f} | {ps/tot:.3f} | {bps/btot:.3f} |")
        # by event index
        bins = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 10**9)]
        row = [reg]
        for lo, hi in bins:
            n = k = 0
            for e in ev:
                for i, (_, p) in enumerate(e):
                    if lo <= i < hi:
                        n += 1; k += int(p)
            row.append(f"{k}/{n} = {k/n:.2f}" if n else "n/a")
        idx_rows.append("| " + " | ".join(row) + " |")
    L += ["", "Pass rate by event index within a run (all events, 0-based index):", "",
          "| regime | event 0 | events 1-2 | events 3-5 | events 6-11 | events 12+ |", "|---|---|---|---|---|---|", *idx_rows, ""]
    text = "\n".join(L)
    print(text)
    if args.write:
        with open(os.path.join(RESULTS_DIR, "event_structure.md"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
