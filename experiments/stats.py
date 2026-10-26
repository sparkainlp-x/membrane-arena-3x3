"""Small statistics helpers: bootstrap, sign-flip test, Holm, Wilson, ECE (NumPy only)."""

from __future__ import annotations

import math

import numpy as np

N_BOOT = 10_000
N_FLIPS = 100_000
BOOT_SEED = 20261025
FLIP_SEED = 20261026


def paired_effect(diff, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Mean paired difference, dz and 95% percentile bootstrap intervals."""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = d[idx]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dzs = np.where(sds > 0, means / sds, np.nan)
    sd = d.std(ddof=1)
    return {
        "n": n,
        "mean": float(d.mean()),
        "mean_ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "dz": float(d.mean() / sd) if sd > 0 else None,
        "dz_ci": [float(np.nanpercentile(dzs, 2.5)), float(np.nanpercentile(dzs, 97.5))]
        if np.isfinite(dzs).any()
        else None,
    }


def sign_flip_p(diff, n_flips: int = N_FLIPS, seed: int = FLIP_SEED) -> float:
    """Two-sided paired sign-flip permutation p = (hits + 1) / (n_flips + 1)."""
    d = np.asarray(diff, dtype=float)
    obs = abs(d.sum())
    rng = np.random.default_rng(seed)
    hits = 0
    chunk = 10_000
    for start in range(0, n_flips, chunk):
        m = min(chunk, n_flips - start)
        signs = rng.choice((-1.0, 1.0), size=(m, len(d)))
        hits += int((np.abs(signs @ d) >= obs - 1e-12).sum())
    return (hits + 1) / (n_flips + 1)


def holm(pvalues: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm step-down: adjusted p-values and reject flags, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return [{"p": pvalues[i], "p_holm": adj[i], "reject": adj[i] < alpha} for i in range(m)]


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]


N_BINS = 5


def _bin_arrays(per_run_forecasts, per_run_outcomes):
    """Per-run bin counts, forecast sums, outcome sums: three (runs, N_BINS) arrays."""
    runs = len(per_run_forecasts)
    cnt = np.zeros((runs, N_BINS)); sp = np.zeros((runs, N_BINS)); sy = np.zeros((runs, N_BINS))
    for r, (fs, ys) in enumerate(zip(per_run_forecasts, per_run_outcomes)):
        for p, y in zip(fs, ys):
            b = min(int(p * N_BINS), N_BINS - 1)
            cnt[r, b] += 1; sp[r, b] += p; sy[r, b] += float(y)
    return cnt, sp, sy


def _ece(cnt, sp, sy) -> float:
    n = cnt.sum()
    if n == 0:
        return float("nan")
    mask = cnt > 0
    return float((np.abs(sp[mask] - sy[mask])).sum() / n)


def ece_with_ci(per_run_forecasts, per_run_outcomes, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Pooled ECE (5 equal-width bins), cluster bootstrap over runs, plus a reliability table."""
    cnt, sp, sy = _bin_arrays(per_run_forecasts, per_run_outcomes)
    point = _ece(cnt.sum(0), sp.sum(0), sy.sum(0))
    rng = np.random.default_rng(seed)
    runs = cnt.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, runs, size=runs)
        vals.append(_ece(cnt[idx].sum(0), sp[idx].sum(0), sy[idx].sum(0)))
    vals = np.array(vals)
    table = []
    for b in range(N_BINS):
        c = cnt[:, b].sum()
        table.append({
            "bin": f"{b / N_BINS:.1f}-{(b + 1) / N_BINS:.1f}",
            "n": int(c),
            "mean_forecast": float(sp[:, b].sum() / c) if c else None,
            "observed_rate": float(sy[:, b].sum() / c) if c else None,
        })
    return {
        "ece": point,
        "ece_ci": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))],
        "reliability": table,
    }
