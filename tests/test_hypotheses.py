"""Falsifiable hypotheses, measured with the benchmark (SYNTHETIC).

Each test states the regime it was measured in. If a hypothesis fails
here, the hypothesis is wrong for that regime — the test is the check,
not a claim that the regime must hold everywhere.
"""

from scripts.benchmark import run_one


def test_small_dt_exact_mode_survives_200_steps_above_90_percent():
    # Measured (seeds 0-4, coupling 0.02, adaptive retry on):
    # dt=0.01, exact mode: 5/5 survived, accepted fraction 1.0.
    rows = [
        run_one(s, 200, dt=0.01, coupling=0.02, operator_mode="exact")
        for s in range(5)
    ]
    assert all(r["survived"] for r in rows)
    assert all(r["accepted_fraction"] > 0.90 for r in rows)
    assert all(r["final_max_residual"] < 0.20 for r in rows)


def test_accepted_fraction_falls_as_dt_grows():
    # Measured (seeds 0-4, coupling 0.02, legacy, 50 steps):
    # dt 0.01 -> 0.949, dt 0.05 -> 0.532, dt 0.10 -> 0.371.
    means = {}
    for dt in (0.01, 0.05, 0.10):
        rows = [
            run_one(s, 50, dt=dt, coupling=0.02, operator_mode="legacy")
            for s in range(5)
        ]
        means[dt] = sum(r["accepted_fraction"] for r in rows) / len(rows)
    assert means[0.01] > means[0.05] > means[0.10]
    assert means[0.01] > 0.90


def test_coupling_has_no_threshold_in_0_to_0_10_at_dt_0_05():
    # Falsified threshold claim, measured (seeds 0-4, dt 0.05, legacy,
    # 50 steps): coupling 0.0, 0.02 and 0.10 give identical step and
    # acceptance counts. The coupling term is inert at this scale
    # (see README Known limit 7); no coupling threshold exists here.
    by_coupling = {}
    for coupling in (0.0, 0.02, 0.10):
        rows = [
            run_one(s, 50, dt=0.05, coupling=coupling, operator_mode="legacy")
            for s in range(5)
        ]
        by_coupling[coupling] = (
            [r["steps_executed"] for r in rows],
            [r["accepted_steps"] for r in rows],
        )
    assert by_coupling[0.0] == by_coupling[0.02] == by_coupling[0.10]


def test_residual_grows_with_dt_in_legacy_mode():
    # Measured final max residual (seeds 0-4, legacy, 50 steps):
    # dt 0.01 -> ~0.179, dt 0.05 -> ~0.198, dt 0.10 -> ~0.194.
    # Residual is already near the 0.20 guard at dt >= 0.05, which is
    # why those runs latch; at dt 0.01 it stays measurably lower.
    def mean_residual(dt):
        rows = [
            run_one(s, 50, dt=dt, coupling=0.02, operator_mode="legacy")
            for s in range(5)
        ]
        return sum(r["final_max_residual"] for r in rows) / len(rows)

    assert mean_residual(0.01) < mean_residual(0.05)
