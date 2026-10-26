"""Pilot: does the intermediate regime run, and roughly what is its pass rate?

Uses pilot seeds 900 to 909 only (disjoint from the dev and confirm seeds of
the pre-registered analysis). The regime itself is fixed by definition in
experiments/regimes.py; this pilot cannot change it. SYNTHETIC.
"""

from __future__ import annotations

import sys

from membrane_arena_3x3.crystallize_injection import InjectionConfig, run_simulation_with_injection

from experiments.regimes import REGIMES

PILOT_SEEDS = range(900, 910)


def main() -> int:
    print("| regime | pilot seeds | mean steps | mean accepted | pass rate | breaker open |")
    print("|---|---|---|---|---|---|")
    for name, reg in REGIMES.items():
        steps = acc = ev = ps = latched = 0
        for seed in PILOT_SEEDS:
            s = run_simulation_with_injection(
                reg.engine_config(seed),
                InjectionConfig(first_factor=reg.first_factor, mirror_variant=reg.mirror_variant),
            )
            steps += s["steps_executed"]; acc += s["accepted_steps"]
            ev += s["injection_events"]; ps += s["injection_passes"]
            latched += s["status"] == "circuit_breaker_open"
        n = len(PILOT_SEEDS)
        print(f"| {name} | {n} | {steps/n:.1f} | {acc/n:.1f} | {ps}/{ev} = {ps/max(ev,1):.2f} | {latched}/{n} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
