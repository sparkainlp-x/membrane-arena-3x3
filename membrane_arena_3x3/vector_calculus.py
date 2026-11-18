# Copyright (c) 2026 Jean-Francois Brisson (Spark AI NLP)
# SPDX-License-Identifier: AGPL-3.0-only
"""Discrete vector calculus on the 3x3 membrane lattice (SYNTHETIC).

The membrane's stock "divergence" is ``1 - fidelity`` between cell pairs:
a scalar dissimilarity, NOT ``div F`` of a vector field. This module adds
a genuine discrete vector calculus on the lattice, built from one scalar
field, and measures whether it predicts anything.

Scalar field
------------
``s[i, j]`` is the fidelity of cell ``(i, j)`` to its reference state
(``kind="fidelity"``), or its purity (``kind="purity"``). The flow field
is the descent field ``F = -grad s``: flow runs from high-fidelity cells
toward low-fidelity ones.

Operators (finite-volume, spacing ``h = 1``)
---------------------------------------------
``F`` lives on cell faces. The east face of cell ``(i, j)`` carries
``Fx = -(s[i, j+1] - s[i, j])``; the north face carries
``Fy = -(s[i+1, j] - s[i, j])``. Boundary faces carry zero flux: the
lattice is closed, so fidelity can only redistribute between cells,
never leave the lattice. The divergence of cell ``(i, j)`` is the net
outward face flux,

    div[i, j] = (Fx_east - Fx_west) + (Fy_north - Fy_south).

The curl is the circulation of ``F`` around each cell, with tangential
edge values averaged from the node-centred field ``-grad s``. Each
interior edge is shared by two cells with opposite orientation, so both
sums below telescope.

Identities (validation tautologies, reported as such)
-----------------------------------------------------
* Gauss: ``sum(div) == boundary outward flux`` exactly (telescoping).
* Stokes: ``sum(curl) == boundary circulation`` exactly (telescoping).
Both hold by construction for ANY scalar field, including random noise;
they validate the implementation, they discover nothing about the
membrane. The real measurement is the escape-predictor claim.

Escape prediction
-----------------
An *escape event* at step ``t`` is a rejected step whose guard reason is
a divergence-band escape: ``pairwise_divergence_below_minimum`` (a
pair collided — the margin guard firing), ``divergence_below_minimum``
(the whole lattice synchronised out of the band) or
``divergence_above_maximum`` (a pair flew apart). Features are computed
from the PRE-step state, so the prediction is genuinely one step ahead:
does the divergence field at step ``t - 1`` predict the escape at ``t``?

The head-to-head baseline is the stock scalar, ``1 - fidelity`` to the
reference (``max_infidelity``), and secondarily the largest per-cell
fidelity drop across the previous step (``max_fidelity_drop``).

Frozen thresholds and claim bars live in
``docs/vector_calculus_preregistration.md`` and are mirrored as the
``FROZEN_*`` constants below. They were frozen on a calibration split
(seeds 42, 1, 7) BEFORE the evaluation split was measured; they are never
moved to force a pass.

Honest limits
-------------
Synthetic lattice toy. The "divergence" here is a diagnostic on a
discrete 3x3 grid of fidelity scores, NOT ``div F`` of a physical field;
no physical field-theory claim is made. Gauss/Stokes are exact
telescoping identities, not discoveries. An escape is a guard rejection
inside this simulator, not a physical event.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import fidelity, purity
from .engine import EngineConfig, MembraneArena3x3
from .lattice import LatticeConfig
from .shields import GuardConfig

# --- Frozen pre-registered thresholds (see docs/vector_calculus_preregistration.md) ---
# Set by calibration on seeds 42/1/7; frozen before the evaluation split.
FROZEN_DIV_FLAG = 0.15
"""Flag a step as high-divergence when max |div| exceeds this."""

FROZEN_FID_FLAG = 0.60
"""Flag a step on the baseline when max (1 - fidelity) exceeds this."""

FROZEN_MIN_PRECISION_LIFT = 2.0
"""Claim (i) needs precision >= this multiple of the escape base rate."""

FROZEN_MIN_RECALL = 0.30
"""Claim (i) needs recall at least this."""

FROZEN_MIN_AUC_GAIN = 0.05
"""Claim (iv) needs AUC(div) - AUC(baseline) at least this."""

#: Guard reasons that count as a divergence-band escape event.
ESCAPE_REASONS = frozenset(
    {
        "pairwise_divergence_below_minimum",
        "divergence_below_minimum",
        "divergence_above_maximum",
    }
)


def scalar_field(
    state: np.ndarray, reference: np.ndarray, kind: str = "fidelity"
) -> np.ndarray:
    """Return the (3, 3) scalar field for a lattice state.

    ``kind="fidelity"``: fidelity of each cell to its reference state.
    ``kind="purity"``: ``tr(rho^2)`` of each cell.
    """
    if kind not in ("fidelity", "purity"):
        raise ValueError("kind must be 'fidelity' or 'purity'")
    cells = state.reshape(-1, state.shape[-2], state.shape[-1])
    refs = reference.reshape(-1, reference.shape[-2], reference.shape[-1])
    if kind == "fidelity":
        values = [fidelity(cells[k], refs[k]) for k in range(9)]
    else:
        values = [purity(cells[k]) for k in range(9)]
    return np.asarray(values, dtype=float).reshape(3, 3)


def node_gradient(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ds/dj, ds/di)`` on nodes: central inside, one-sided at edges."""
    s = np.asarray(s, dtype=float)
    if s.shape != (3, 3):
        raise ValueError("scalar field must be 3x3")
    gx = np.empty_like(s)  # along j (columns, "east")
    gy = np.empty_like(s)  # along i (rows, "north")
    gx[:, 1] = (s[:, 2] - s[:, 0]) / 2.0
    gx[:, 0] = s[:, 1] - s[:, 0]
    gx[:, 2] = s[:, 2] - s[:, 1]
    gy[1, :] = (s[2, :] - s[0, :]) / 2.0
    gy[0, :] = s[1, :] - s[0, :]
    gy[2, :] = s[2, :] - s[1, :]
    return gx, gy


def edge_fluxes(s: np.ndarray) -> dict[str, np.ndarray]:
    """Return the flow ``F = -grad s`` on every cell face.

    ``fx_east[i, j]`` is the x-component through the east face of cell
    ``(i, j)``; ``fx_west`` the west face; ``fy_north`` / ``fy_south``
    likewise. Interior faces are the negative neighbour difference;
    boundary faces are zero (closed lattice: no flux leaves).
    """
    s = np.asarray(s, dtype=float)
    if s.shape != (3, 3):
        raise ValueError("scalar field must be 3x3")
    fx_east = np.empty((3, 3))
    fx_east[:, 0] = -(s[:, 1] - s[:, 0])
    fx_east[:, 1] = -(s[:, 2] - s[:, 1])
    fx_east[:, 2] = 0.0  # boundary: closed lattice, zero flux
    fx_west = np.empty((3, 3))
    fx_west[:, 0] = 0.0  # boundary: closed lattice, zero flux
    fx_west[:, 1] = -(s[:, 1] - s[:, 0])
    fx_west[:, 2] = -(s[:, 2] - s[:, 1])
    fy_north = np.empty((3, 3))
    fy_north[0, :] = -(s[1, :] - s[0, :])
    fy_north[1, :] = -(s[2, :] - s[1, :])
    fy_north[2, :] = 0.0  # boundary
    fy_south = np.empty((3, 3))
    fy_south[0, :] = 0.0  # boundary
    fy_south[1, :] = -(s[1, :] - s[0, :])
    fy_south[2, :] = -(s[2, :] - s[1, :])
    return {
        "fx_east": fx_east,
        "fx_west": fx_west,
        "fy_north": fy_north,
        "fy_south": fy_south,
    }


def divergence(s: np.ndarray) -> np.ndarray:
    """Return the (3, 3) finite-volume divergence of ``-grad s``."""
    faces = edge_fluxes(s)
    return (
        (faces["fx_east"] - faces["fx_west"])
        + (faces["fy_north"] - faces["fy_south"])
    )


def gauss_identity(s: np.ndarray) -> dict[str, float]:
    """Check Gauss on the lattice: ``sum(div)`` vs boundary outward flux.

    Returns total divergence, boundary flux and their residual. The
    lattice is closed (zero boundary flux), so both are zero up to
    floating-point rounding for any field: the interior face fluxes
    cancel pairwise (telescoping). This is a validation tautology, not
    a discovery — but a wrong sign or index in any interior face would
    break it, so it does pin the implementation.
    """
    s = np.asarray(s, dtype=float)
    div = divergence(s)
    faces = edge_fluxes(s)
    boundary_flux = float(
        np.sum(faces["fx_east"][:, 2])
        - np.sum(faces["fx_west"][:, 0])
        + np.sum(faces["fy_north"][2, :])
        - np.sum(faces["fy_south"][0, :])
    )
    total = float(np.sum(div))
    return {
        "total_divergence": total,
        "boundary_flux": boundary_flux,
        "residual": total - boundary_flux,
    }


def circulation(s: np.ndarray) -> np.ndarray:
    """Return the (3, 3) circulation of ``F = -grad s`` around each cell.

    Tangential edge values are averages of the node-centred field over
    the two cells sharing the face: the top/bottom faces of a cell use
    the x-component averaged across the face (neighbours in ``i``), the
    left/right faces the y-component averaged across the face
    (neighbours in ``j``). Each interior face is shared by two cells
    with opposite orientation, so the sum over cells telescopes to the
    boundary circulation.
    """
    s = np.asarray(s, dtype=float)
    gx, gy = node_gradient(s)
    fx = -gx  # node-centred flow, x-component
    fy = -gy  # node-centred flow, y-component
    # Face values: average of the two adjacent cell centres; a boundary
    # face takes the single adjacent centre's value.
    bottom = np.empty((3, 3))  # south face: Fx averaged across i
    top = np.empty((3, 3))  # north face: Fx averaged across i
    left = np.empty((3, 3))  # west face: Fy averaged across j
    right = np.empty((3, 3))  # east face: Fy averaged across j
    for i in range(3):
        for j in range(3):
            bottom[i, j] = fx[i, j] if i == 0 else 0.5 * (fx[i - 1, j] + fx[i, j])
            top[i, j] = fx[i, j] if i == 2 else 0.5 * (fx[i, j] + fx[i + 1, j])
            left[i, j] = fy[i, j] if j == 0 else 0.5 * (fy[i, j - 1] + fy[i, j])
            right[i, j] = fy[i, j] if j == 2 else 0.5 * (fy[i, j] + fy[i, j + 1])
    # Counter-clockwise: bottom left->right, right bottom->top,
    # top right->left, left top->bottom.
    return (bottom - top) + (right - left)


def stokes_identity(s: np.ndarray) -> dict[str, float]:
    """Check Stokes on the lattice: ``sum(curl)`` vs boundary circulation.

    Returns total circulation, boundary circulation and their residual.
    The residual is zero up to floating-point rounding for any field:
    interior edges cancel pairwise (telescoping). This is a validation
    tautology, not a discovery.
    """
    s = np.asarray(s, dtype=float)
    curl = circulation(s)
    gx, gy = node_gradient(s)
    fx = -gx
    fy = -gy
    # Boundary circulation, counter-clockwise: bottom row west->east
    # (fx on row 0), east column south->north (fy on column 2), top row
    # east->west (-fx on row 2), west column north->south (-fy on col 0).
    boundary = float(
        np.sum(fx[0, :])
        + np.sum(fy[:, 2])
        - np.sum(fx[2, :])
        - np.sum(fy[:, 0])
    )
    total = float(np.sum(curl))
    return {
        "total_circulation": total,
        "boundary_circulation": boundary,
        "residual": total - boundary,
    }


def step_features(
    state: np.ndarray,
    reference: np.ndarray,
    previous_state: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute one-step-ahead predictor features from a pre-step state.

    ``max_abs_div`` / ``max_div`` come from the discrete divergence of
    the fidelity field. Baselines: ``max_infidelity`` is the stock
    scalar ``max(1 - fidelity)``; ``max_fidelity_drop`` is the largest
    per-cell fidelity fall since ``previous_state`` (0.0 when unknown).
    """
    s = scalar_field(state, reference, kind="fidelity")
    div = divergence(s)
    infidelity = 1.0 - s
    features = {
        "max_abs_div": float(np.max(np.abs(div))),
        "max_div": float(np.max(div)),
        "mean_abs_div": float(np.mean(np.abs(div))),
        "max_infidelity": float(np.max(infidelity)),
        "mean_infidelity": float(np.mean(infidelity)),
        "max_fidelity_drop": 0.0,
    }
    if previous_state is not None:
        s_prev = scalar_field(previous_state, reference, kind="fidelity")
        features["max_fidelity_drop"] = float(np.max(s_prev - s))
    return features


@dataclass
class EscapeRecord:
    """One labelled step: pre-step features + whether the step escaped."""

    seed: int
    config_name: str
    step: int
    escaped: bool
    rejected: bool
    reason: str
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "config_name": self.config_name,
            "step": self.step,
            "escaped": self.escaped,
            "rejected": self.rejected,
            "reason": self.reason,
            "features": self.features,
        }


def collect_escape_dataset(
    configs: dict[str, EngineConfig],
    seeds: list[int],
    steps: int = 200,
) -> list[EscapeRecord]:
    """Run each config/seed and label every step with pre-step features.

    Each run steps the arena manually (``step()``, not ``run()``) so the
    pre-step state is visible before the outcome is known. A step is
    labelled ``escaped`` when it is rejected with a divergence-band
    guard reason (see ``ESCAPE_REASONS``). Runs stop when the breaker
    latches, exactly as the stock engine stops.
    """
    records: list[EscapeRecord] = []
    for name, config in configs.items():
        for seed in seeds:
            cfg = EngineConfig(
                lattice=config.lattice,
                guards=config.guards,
                steps=steps,
                seed=seed,
                schumann=config.schumann,
                schumann_hz=config.schumann_hz,
                schumann_amplitude=config.schumann_amplitude,
                failure_limit=config.failure_limit,
                adaptive_retry=config.adaptive_retry,
                max_retries=config.max_retries,
                retry_dt_factors=config.retry_dt_factors,
                retry_mirror_factors=config.retry_mirror_factors,
                retry_coupling_factors=config.retry_coupling_factors,
                retry_policy=config.retry_policy,
                drive_time_mode=config.drive_time_mode,
            )
            arena = MembraneArena3x3(cfg)
            if name == "planted_collision":
                # Two cells start identical: a collision the pairwise
                # guard must catch. (Harness-side initial condition;
                # the engine and guards are untouched.)
                arena.lattice.state[0, 1] = arena.lattice.state[0, 0].copy()
            previous_state: np.ndarray | None = None
            for _ in range(steps):
                if arena.circuit_breaker:
                    break
                features = step_features(
                    arena.lattice.state, arena.lattice.reference, previous_state
                )
                previous_state = arena.lattice.state.copy()
                result = arena.step()
                reason = result.get("final_step_reason", result.get("reason", ""))
                rejected = result["status"] == "rejected"
                records.append(
                    EscapeRecord(
                        seed=seed,
                        config_name=name,
                        step=result["step"],
                        escaped=bool(rejected and reason in ESCAPE_REASONS),
                        rejected=rejected,
                        reason=reason,
                        features=features,
                    )
                )
    return records


def auc_rank(scores: list[float], labels: list[bool]) -> float:
    """Rank (Mann-Whitney) AUC of ``scores`` against boolean ``labels``.

    0.5 is chance; 1.0 is perfect separation. Returns 0.5 when a class
    is absent. No third-party dependency.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(np.sum(labels))
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    _, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True
    )
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    avg_ranks = sums / counts
    ranks = avg_ranks[inverse]
    rank_pos = float(np.sum(ranks[labels]))
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate_predictor(
    records: list[EscapeRecord],
    div_flag: float = FROZEN_DIV_FLAG,
    fid_flag: float = FROZEN_FID_FLAG,
) -> dict:
    """Score divergence-as-escape-predictor against the fidelity baseline.

    Flags use the FROZEN thresholds. Returns precision/recall/F1 for the
    divergence flag, the escape base rate, and rank AUCs for the
    divergence score, the ``1 - fidelity`` baseline and the fidelity-drop
    baseline, all on the same labelled steps.
    """
    if not records:
        raise ValueError("no records to evaluate")
    labels = [r.escaped for r in records]
    n_escapes = sum(labels)
    base_rate = n_escapes / len(records)
    div_scores = [r.features["max_abs_div"] for r in records]
    fid_scores = [r.features["max_infidelity"] for r in records]
    drop_scores = [r.features["max_fidelity_drop"] for r in records]
    flagged = [s > div_flag for s in div_scores]
    tp = sum(1 for f, y in zip(flagged, labels) if f and y)
    fp = sum(1 for f, y in zip(flagged, labels) if f and not y)
    fn = sum(1 for f, y in zip(flagged, labels) if not f and y)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fid_flagged = [s > fid_flag for s in fid_scores]
    fid_tp = sum(1 for f, y in zip(fid_flagged, labels) if f and y)
    fid_fp = sum(1 for f, y in zip(fid_flagged, labels) if f and not y)
    fid_fn = sum(1 for f, y in zip(fid_flagged, labels) if not f and y)
    fid_precision = fid_tp / (fid_tp + fid_fp) if (fid_tp + fid_fp) else 0.0
    fid_recall = fid_tp / (fid_tp + fid_fn) if (fid_tp + fid_fn) else 0.0
    auc_div = auc_rank(div_scores, labels)
    auc_fid = auc_rank(fid_scores, labels)
    auc_drop = auc_rank(drop_scores, labels)
    return {
        "n_steps": len(records),
        "n_escapes": n_escapes,
        "base_rate": base_rate,
        "div_flag": div_flag,
        "fid_flag": fid_flag,
        "div_precision": precision,
        "div_recall": recall,
        "div_f1": f1,
        "div_flagged": int(sum(flagged)),
        "div_tp": tp,
        "div_fp": fp,
        "div_fn": fn,
        "precision_lift": (precision / base_rate) if base_rate else 0.0,
        "fid_precision": fid_precision,
        "fid_recall": fid_recall,
        "auc_div": auc_div,
        "auc_fid": auc_fid,
        "auc_drop": auc_drop,
        "auc_gain_vs_fid": auc_div - auc_fid,
        "auc_gain_vs_drop": auc_div - auc_drop,
        "claim_i_pass": bool(
            precision >= FROZEN_MIN_PRECISION_LIFT * base_rate
            and recall >= FROZEN_MIN_RECALL
        ),
        "claim_iv_pass": bool((auc_div - auc_fid) >= FROZEN_MIN_AUC_GAIN),
    }


def standard_configs() -> dict[str, EngineConfig]:
    """The config mix used for the escape-predictor measurement.

    Stock runs rarely escape on a divergence guard, so the mix adds
    regimes that do: an offset ladder (desynchronised clocks collide
    pairs), a planted collision, exact mode (lattice-wide synchronisation
    exits the band from below) and high coupling.
    """
    stock = EngineConfig(steps=200, seed=42)
    ladder = EngineConfig(
        steps=200,
        seed=42,
        lattice=LatticeConfig(
            dt_multipliers=(1.0, 1.5, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        ),
    )
    exact = EngineConfig(
        steps=200, seed=42, lattice=LatticeConfig(operator_mode="exact")
    )
    planted = EngineConfig(steps=200, seed=42)
    high_coupling = EngineConfig(
        steps=200, seed=42, lattice=LatticeConfig(coupling=0.10)
    )
    return {
        "stock": stock,
        "offset_ladder": ladder,
        "exact": exact,
        "planted_collision": planted,
        "high_coupling": high_coupling,
    }
