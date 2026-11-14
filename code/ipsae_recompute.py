"""ipSAE from an AF3 confidences JSON, plus the other confidence-head readouts.

ipSAE (Dunbrack, bioRxiv 2025.02.10.637595) applies a pTM-style transformation
to the inter-chain block of PAE under a 10 A error cutoff.  For a direction
c1 -> c2, a pair (i, j) with i in c1 and j in c2 is valid if PAE(i, j) < 10 A,
and with n0i the number of valid partners of residue i (equation 1):

    ipSAE_{c1->c2} = max_i  (1/n0i) * sum_{j valid} 1 / (1 + (PAE(i,j)/d0(n0i))^2)

    d0(n) = max(1.0, 1.24 * (max(n, 26) - 15)^(1/3) - 1.8)

We rank by ipSAE_min, the lower of the two directions, and also report
ipSAE_max (which matches the source study).  ipSAE is zero when no residue pair
across the interface passes the cutoff — an *absent* prediction, not a weak one,
which is what 71.4% of the no-MSA folds produce (section 4.1).

Validated in E0 against the deposit's independently computed values before any
production run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAE_CUTOFF = 10.0


def d0(n: float) -> float:
    return max(1.0, 1.24 * (max(float(n), 26.0) - 15.0) ** (1.0 / 3.0) - 1.8)


def ipsae_directional(pae: np.ndarray, idx1: np.ndarray, idx2: np.ndarray,
                      cutoff: float = PAE_CUTOFF) -> float:
    """max over residues i in c1 of the n0i-normalised pTM-style sum."""
    if len(idx1) == 0 or len(idx2) == 0:
        return 0.0
    sub = pae[np.ix_(idx1, idx2)]                  # PAE(i in c1, j in c2)
    valid = sub < cutoff
    n0 = valid.sum(axis=1)
    rows = np.flatnonzero(n0 > 0)
    if rows.size == 0:
        return 0.0
    best = 0.0
    for i in rows:
        n = int(n0[i])
        d = d0(n)
        vals = sub[i][valid[i]]
        best = max(best, float(np.sum(1.0 / (1.0 + (vals / d) ** 2)) / n))
    return best


def chain_indices(token_chain_ids: list[str]) -> dict:
    arr = np.asarray(token_chain_ids)
    return {c: np.flatnonzero(arr == c) for c in dict.fromkeys(token_chain_ids)}


def load_confidences(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def score_fold(conf_path: str | Path, summary_path: str | Path | None = None,
               chain1: str = "A", chain2: str = "B",
               cutoff: float = PAE_CUTOFF) -> dict:
    """Every scalar we extract from one fold's confidence output.

    chain1 is the target, chain2 the binder.  Coordinates are never read:
    these are measurements of the confidence head (section 2.1).
    """
    conf = load_confidences(conf_path)
    pae = np.asarray(conf["pae"], dtype=float)
    tci = conf.get("token_chain_ids")
    if tci is None:
        raise KeyError(f"{conf_path}: no token_chain_ids")
    idx = chain_indices(tci)
    if chain1 not in idx or chain2 not in idx:
        raise KeyError(f"{conf_path}: chains {chain1}/{chain2} not in {list(idx)}")
    i1, i2 = idx[chain1], idx[chain2]

    ab = ipsae_directional(pae, i1, i2, cutoff)
    ba = ipsae_directional(pae, i2, i1, cutoff)
    inter = pae[np.ix_(i1, i2)]
    below = inter < cutoff

    out = {
        "ipsae_min": min(ab, ba),
        "ipsae_max": max(ab, ba),
        "ipsae_mean": 0.5 * (ab + ba),
        "ipsae_a_to_b": ab,
        "ipsae_b_to_a": ba,
        "n_target_tokens": int(len(i1)),
        "n_binder_tokens": int(len(i2)),
        # interface PAE summaries
        "ipae_mean": float(inter.mean()),
        "ipae_min": float(inter.min()),
        "ipae_frac_below_cutoff": float(below.mean()),
        "ipae_n_below_cutoff": int(below.sum()),
    }

    # interface contact probabilities, when present
    cp = conf.get("contact_probs")
    if cp is not None:
        cp = np.asarray(cp, dtype=float)[np.ix_(i1, i2)]
        out["contact_prob_max"] = float(cp.max())
        out["contact_prob_sum"] = float(cp.sum())

    # pLDDT summaries, whole complex and per chain
    plddt = conf.get("atom_plddts")
    acid = conf.get("atom_chain_ids")
    if plddt is not None:
        plddt = np.asarray(plddt, dtype=float)
        out["plddt_mean"] = float(plddt.mean())
        if acid is not None:
            acid = np.asarray(acid)
            for name, c in (("target", chain1), ("binder", chain2)):
                m = acid == c
                if m.any():
                    out[f"plddt_mean_{name}"] = float(plddt[m].mean())
                    out[f"plddt_min_{name}"] = float(plddt[m].min())

    if summary_path and Path(summary_path).exists():
        s = json.loads(Path(summary_path).read_text())
        for k in ("iptm", "ptm", "ranking_score", "fraction_disordered", "has_clash"):
            if k in s and s[k] is not None:
                out[k] = float(s[k])
        cpi = s.get("chain_pair_iptm")
        if cpi is not None:
            a = np.asarray(cpi, dtype=float)
            if a.ndim == 2 and a.shape[0] >= 2:
                out["chain_pair_iptm"] = float(a[0, 1])
        cpp = s.get("chain_pair_pae_min")
        if cpp is not None:
            a = np.asarray(cpp, dtype=float)
            if a.ndim == 2 and a.shape[0] >= 2:
                out["chain_pair_pae_min"] = float(a[0, 1])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("confidences", help="path to a *_confidences.json")
    p.add_argument("--summary", default=None)
    p.add_argument("--chain1", default="A", help="target chain")
    p.add_argument("--chain2", default="B", help="binder chain")
    p.add_argument("--cutoff", type=float, default=PAE_CUTOFF)
    a = p.parse_args()
    summary = a.summary
    if summary is None:
        guess = Path(a.confidences.replace("_confidences.json",
                                           "_summary_confidences.json"))
        summary = str(guess) if guess.exists() else None
    print(json.dumps(score_fold(a.confidences, summary, a.chain1, a.chain2,
                                a.cutoff), indent=2))


if __name__ == "__main__":
    main()
