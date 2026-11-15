"""
Metrics for the SC4001 binder re-ranking experiments.

Every definition here mirrors the one written in the report, so that the
re-ranker numbers are directly comparable to the ablation numbers:

  * average precision (AP)      -- Draft.tex, "Scoring", AP equation
  * AUC-ROC via the rank sum    -- Draft.tex, "Scoring", AUC-ROC equation
  * target-level paired bootstrap, 2,000 resamples

Two details matter and are easy to get wrong.

1. TIES.  The deposited ipSAE scores are rounded to three decimals (only ~740
   distinct values over 3,669 designs), and without the MSA 71.4% of them are
   exactly 0.000.  A deterministic sort would make AP depend on the row order
   of the CSV, which is meaningless.  Every AP below is therefore averaged over
   `n_tiebreaks` independent random orderings of tied scores -- the same
   protocol used for the ablation results in the report.

2. GROUPING.  The target (receptor) is the unit of replication.  AP is computed
   *inside* each target and then averaged over targets.  Designs for one target
   are not independent replicates, so a pooled AP would overstate precision.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ap_from_order",
    "average_precision",
    "roc_auc_ranksum",
    "per_target_scores",
    "paired_bootstrap",
]


# --------------------------------------------------------------------------
# Average precision
# --------------------------------------------------------------------------

def ap_from_order(y_sorted: np.ndarray) -> float:
    """Average precision given labels already sorted best-first.

    Implements the formula in the report directly:

        AP = (1 / P) * sum_{i : y_i = 1} tp(i) / i

    where i is the 1-indexed rank, tp(i) is the number of positives at or above
    rank i, and P is the total number of positives.

    Returns NaN when the group contains no positives (AP is undefined there).
    """
    y_sorted = np.asarray(y_sorted, dtype=np.int64)
    n_pos = int(y_sorted.sum())
    if n_pos == 0:
        return float("nan")

    ranks = np.arange(1, y_sorted.size + 1, dtype=np.float64)
    tp = np.cumsum(y_sorted, dtype=np.float64)
    hit = y_sorted == 1
    return float((tp[hit] / ranks[hit]).sum() / n_pos)


def average_precision(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_tiebreaks: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """AP with random tie-breaking, averaged over `n_tiebreaks` draws.

    Ties are broken by a fresh uniform random key on each draw, so the returned
    value is the *expected* AP over orderings rather than an artefact of the
    input order.  With continuous scores (e.g. an MLP output) there are no ties
    and all draws agree, so this costs a little time but changes nothing.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.size != scores.size:
        raise ValueError("y_true and scores must have the same length")
    if y_true.sum() == 0:
        return float("nan")

    rng = np.random.default_rng() if rng is None else rng

    # Fast path: no ties means every tie-break draw gives the same ordering.
    if np.unique(scores).size == scores.size:
        order = np.argsort(-scores, kind="stable")
        return ap_from_order(y_true[order])

    total = 0.0
    for _ in range(n_tiebreaks):
        jitter = rng.random(scores.size)
        # lexsort uses the LAST key as primary, so -scores dominates and the
        # random key only resolves exact ties.
        order = np.lexsort((jitter, -scores))
        total += ap_from_order(y_true[order])
    return total / n_tiebreaks


def ap_tiebreak_sd(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_tiebreaks: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Standard deviation of AP across tie-break draws.

    The report quotes this as a diagnostic (tie s.d. 0.018 without the MSA
    versus 0.001 with it).  A large value means the score is so degenerate that
    the ranking is mostly arbitrary.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.sum() == 0:
        return float("nan")
    rng = np.random.default_rng() if rng is None else rng
    vals = []
    for _ in range(n_tiebreaks):
        jitter = rng.random(scores.size)
        order = np.lexsort((jitter, -scores))
        vals.append(ap_from_order(y_true[order]))
    return float(np.std(vals))


# --------------------------------------------------------------------------
# AUC-ROC
# --------------------------------------------------------------------------

def roc_auc_ranksum(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC-ROC by the rank-sum formulation used in the report:

        AUC = (S - P(P+1)/2) / (P * N)

    with S the sum of the positives' ranks, P the positive count and N the
    negative count.  Tied scores share their average rank, which makes a block
    of identical scores contribute exactly 0.5 -- essential here, because a
    degenerate no-MSA prediction ties most of the set at zero.

    Returns NaN if either class is absent.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average ranks (ascending), ties shared.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-indexed average rank
        i = j + 1

    rank_sum = ranks[y_true == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# Per-target aggregation and the paired bootstrap
# --------------------------------------------------------------------------

def per_target_scores(
    targets: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    min_positives: int = 1,
    n_tiebreaks: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """AP and AUC-ROC for every target, keyed by target name.

    `min_positives` drops targets too small to support a reliable AP.  The
    report uses >= 5 confirmed binders for per-receptor analyses (11 of 13
    targets qualify); the headline mean uses all targets with >= 1.
    """
    targets = np.asarray(targets)
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(sorted(set(targets.tolist()))):
        mask = targets == name
        n_pos = int(y_true[mask].sum())
        if n_pos < min_positives:
            continue
        # A per-target seed keeps results reproducible but decorrelated.
        rng = np.random.default_rng(seed * 10_000 + idx)
        out[name] = {
            "n": int(mask.sum()),
            "n_pos": n_pos,
            "ap": average_precision(y_true[mask], scores[mask], n_tiebreaks, rng),
            "auc": roc_auc_ranksum(y_true[mask], scores[mask]),
        }
    return out


def paired_bootstrap(
    ap_a: dict[str, float],
    ap_b: dict[str, float],
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Target-level paired bootstrap of mean(AP_b) - mean(AP_a).

    Resamples *targets* with replacement -- not designs -- because the target is
    the unit of replication.  Both methods are evaluated on the same resampled
    targets, so the comparison stays paired and the between-target variance that
    dominates the data cancels out.

    Returns (point_estimate, ci_low, ci_high).
    """
    names = sorted(set(ap_a) & set(ap_b))
    if not names:
        raise ValueError("no targets in common between the two score dicts")

    a = np.array([ap_a[k] for k in names], dtype=np.float64)
    b = np.array([ap_b[k] for k in names], dtype=np.float64)
    point = float(np.nanmean(b) - np.nanmean(a))

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        pick = rng.integers(0, len(names), len(names))
        deltas[r] = np.nanmean(b[pick]) - np.nanmean(a[pick])

    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return point, lo, hi
