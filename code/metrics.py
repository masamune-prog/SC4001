"""AP, ROC-AUC, tie-breaking and the target-level paired bootstrap.

Everything is computed *within a target* and then averaged over targets: the
target is the unit of replication (section 3.1), because candidates for one
receptor are not independent replicates.

Tie handling is load-bearing here.  Without the MSA, 71.4% of designs score
exactly 0.000 and the deposited scores are rounded to three decimals, so AP is
reported as the mean over 200 independent random tie-breaks (section 3.6).
"""

from __future__ import annotations

import numpy as np

N_TIEBREAKS = 200
N_BOOTSTRAP = 2000


# --- single-ranking metrics ---------------------------------------------------

def average_precision(y: np.ndarray, order: np.ndarray) -> float:
    """AP = (1/P) * sum over ranks holding a positive of tp(i)/i."""
    y_sorted = y[order]
    P = int(y_sorted.sum())
    if P == 0:
        return float("nan")
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    return float((tp[y_sorted == 1] / ranks[y_sorted == 1]).sum() / P)


def ap_tiebreak(y: np.ndarray, s: np.ndarray, n_draws: int = N_TIEBREAKS,
                seed: int = 0) -> tuple[float, float]:
    """Mean and s.d. of AP over `n_draws` random orderings of tied scores."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if np.isnan(s).any():
        keep = ~np.isnan(s)
        y, s = y[keep], s[keep]
    if y.sum() == 0 or len(y) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_draws)
    for d in range(n_draws):
        jitter = rng.random(len(s))
        order = np.lexsort((jitter, -s))      # descending score, ties shuffled
        vals[d] = average_precision(y, order)
    return float(vals.mean()), float(vals.std(ddof=1)) if n_draws > 1 else 0.0


def _rankdata_average(s: np.ndarray) -> np.ndarray:
    """Ranks 1..n ascending, tied values sharing their average rank."""
    n = len(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-sum form; tied scores contribute 0.5 independent of input order."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    P, N = int(y.sum()), int((1 - y).sum())
    if P == 0 or N == 0:
        return float("nan")
    r = _rankdata_average(s)
    S = r[y == 1].sum()
    return float((S - P * (P + 1) / 2) / (P * N))


# --- per-target aggregation ---------------------------------------------------

def per_target(df, score_col: str, label_col: str = "label",
               target_col: str = "target", n_draws: int = N_TIEBREAKS,
               seed: int = 0, min_positives: int = 1) -> "dict[str, dict]":
    out = {}
    for t, g in df.groupby(target_col, sort=True):
        y = g[label_col].to_numpy(dtype=int)
        s = g[score_col].to_numpy(dtype=float)
        if y.sum() < min_positives:
            continue
        ap, sd = ap_tiebreak(y, s, n_draws=n_draws,
                             seed=abs(hash((t, seed))) % (2**31))
        out[t] = {"ap": ap, "ap_tie_sd": sd, "auc": roc_auc(y, s),
                  "n": len(y), "positives": int(y.sum()),
                  "zero_frac": float((s == 0).mean())}
    return out


def mean_ap(pt: dict) -> float:
    return float(np.nanmean([v["ap"] for v in pt.values()]))


def mean_auc(pt: dict) -> float:
    return float(np.nanmean([v["auc"] for v in pt.values()]))


# --- inference ----------------------------------------------------------------

def paired_bootstrap(a: dict, b: dict, key: str = "ap",
                     n_boot: int = N_BOOTSTRAP, seed: int = 0,
                     restrict: list[str] | None = None):
    """Target-level paired bootstrap of mean(b) - mean(a).

    Resampling targets — not designs — is what makes the interval honest given
    the grouped structure of the data.
    """
    targets = sorted(set(a) & set(b))
    if restrict is not None:
        targets = [t for t in targets if t in set(restrict)]
    if not targets:
        raise ValueError("no targets in common")
    da = np.array([a[t][key] for t in targets])
    db = np.array([b[t][key] for t in targets])
    delta = db - da
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(targets), size=(n_boot, len(targets)))
    boots = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "targets": targets,
        "delta": float(delta.mean()),
        "ci": (float(lo), float(hi)),
        "per_target": dict(zip(targets, delta.tolist())),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata_average(x), _rankdata_average(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


# --- self-test ----------------------------------------------------------------

def selftest() -> None:
    """Validation against synthetic data with known ground truth.

    The three cases are those named in the AI Use Statement: only positives
    move, only negatives move, nothing moves.
    """
    # worked AP example from section B: positives at ranks 1, 3, 6 of 10
    y = np.zeros(10, dtype=int); y[[0, 2, 5]] = 1
    s = np.arange(10, 0, -1, dtype=float)
    ap = average_precision(y, np.argsort(-s, kind="mergesort"))
    assert abs(ap - (1 + 2/3 + 1/2) / 3) < 1e-12, ap

    # all-tied scores: AUC exactly 0.5, AP equal to the positive rate in expectation
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.zeros(10)
    assert abs(roc_auc(y, s) - 0.5) < 1e-12
    ap_t, _ = ap_tiebreak(y, s, n_draws=4000, seed=1)
    assert abs(ap_t - 0.2) < 0.02, ap_t

    # perfect and inverted rankings
    y = np.array([1, 1, 0, 0, 0, 0])
    assert roc_auc(y, np.array([5., 4, 3, 2, 1, 0])) == 1.0
    assert roc_auc(y, np.array([0., 1, 2, 3, 4, 5])) == 0.0

    # directional sanity: moving only positives up must raise AP; only negatives
    # up must lower it; moving nothing must leave it unchanged.
    rng = np.random.default_rng(0)
    y = np.array([1] * 20 + [0] * 180)
    base = rng.normal(size=200)
    a0, _ = ap_tiebreak(y, base, n_draws=50, seed=2)
    pos_up = base.copy(); pos_up[y == 1] += 3
    neg_up = base.copy(); neg_up[y == 0] += 3
    a1, _ = ap_tiebreak(y, pos_up, n_draws=50, seed=2)
    a2, _ = ap_tiebreak(y, neg_up, n_draws=50, seed=2)
    a3, _ = ap_tiebreak(y, base.copy(), n_draws=50, seed=2)
    assert a1 > a0 > a2, (a0, a1, a2)
    assert abs(a3 - a0) < 1e-12

    # paired bootstrap recovers a known shift and brackets zero for no shift
    A = {f"t{i}": {"ap": 0.5} for i in range(13)}
    B = {f"t{i}": {"ap": 0.5 + 0.1} for i in range(13)}
    r = paired_bootstrap(A, B, n_boot=500, seed=0)
    assert abs(r["delta"] - 0.1) < 1e-9 and r["excludes_zero"]
    r0 = paired_bootstrap(A, A, n_boot=500, seed=0)
    assert r0["delta"] == 0.0 and not r0["excludes_zero"]

    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    print("metrics selftest: OK")


if __name__ == "__main__":
    selftest()
