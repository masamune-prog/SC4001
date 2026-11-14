"""ipSAE recomputed from the deposit's raw AF3 PAE matrices.

Two jobs:

  E0 validation gate.  Our ipSAE implementation is checked against the
  deposit's independently computed values before any production run.  Run with
  --gate: it prints Spearman/Pearson agreement and the worst disagreements and
  exits non-zero if agreement falls below --gate-rho.

  Feature table.  Writes the 21 AF3 outputs per design, which the Study B
  re-ranker consumes together with the 16 design-side features from the
  deposit.  Produced from the deposit's own folds rather than ours, which is
  why table 4's absolute values are not comparable with table 2 (section 6).

Expected layout, after
    tar --zstd -xf AF3_outputs.tar.zstd --wildcards \
        --no-wildcards-match-slash 'AF3_outputs/*/*_confidences.json'

    AF3_outputs/<design_id>/<something>_confidences.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from af3_io import find_confidences
from dataset import load_dataset
from ipsae_recompute import PAE_CUTOFF, score_fold
from metrics import spearman

AF3_FEATURES = [
    "ipsae_min", "ipsae_max", "ipsae_mean", "iptm", "ptm", "ranking_score",
    "fraction_disordered", "plddt_mean", "plddt_mean_target", "plddt_min_target",
    "plddt_mean_binder", "plddt_min_binder", "ipae_mean", "ipae_min",
    "ipae_frac_below_cutoff", "ipae_n_below_cutoff", "contact_prob_max",
    "contact_prob_sum", "chain_pair_iptm", "chain_pair_pae_min",
    "n_binder_tokens",
]


def design_id_from_path(conf: Path) -> str:
    """The deposit names a directory per design; strip AF3's own suffixes."""
    name = conf.parent.name
    name = re.sub(r"^(seed-\d+_sample-\d+)$", "", name) or conf.parent.parent.name
    return name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--af3-dir", default=".", help="root holding AF3_outputs/")
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--out", default="features_folds.csv")
    # score_fold treats chain1 as the target and chain2 as the binder, and
    # labels its plddt_*_target / plddt_*_binder / n_*_tokens outputs from that.
    #
    # These are the DEPOSIT's folds, whose chain order is the opposite of this
    # project's own (af3_io writes target=A, binder=B).  In the deposit
    # binder_chain == "A" and target_chains == ["B"]; checked against the folds
    # themselves, chain A's token count equals the CSV's A_length (the binder)
    # and chain B's equals B_length (the target) for every single-target design.
    # So here the target is B and the binder is A.
    #
    # The E0 gate itself is unaffected either way: ipsae_min/max/mean are
    # symmetric in the two chains, since swapping them only swaps the two
    # directional scores the min/max/mean are taken over.  Verified on the
    # deposit's folds -- the ipSAE values are bit-identical under the swap.
    #
    # Everything else moved.  plddt_*_target and plddt_*_binder and
    # n_*_tokens were simply exchanged, and the ipae_* block changed value
    # outright, because PAE is NOT symmetric: pae[i,j] != pae[j,i], so the
    # inter-chain block and its transpose have different means and minima.
    # Five of these (plddt_mean/min_target, plddt_mean/min_binder,
    # n_binder_tokens) plus the four ipae_* entries are in AF3_FEATURES.
    p.add_argument("--chain1", default="B", help="target chain (deposit: B)")
    p.add_argument("--chain2", default="A", help="binder chain (deposit: A)")
    p.add_argument("--cutoff", type=float, default=PAE_CUTOFF)
    p.add_argument("--gate", action="store_true",
                   help="E0: validate against the deposit's own ipSAE values")
    p.add_argument("--gate-rho", type=float, default=0.99)
    a = p.parse_args()

    rows = []
    for conf, summary in find_confidences(a.af3_dir):
        did = design_id_from_path(conf)
        try:
            sc = score_fold(conf, summary, a.chain1, a.chain2, a.cutoff)
        except Exception as exc:                            # noqa: BLE001
            print(f"[recompute] FAILED {conf}: {exc}", file=sys.stderr)
            continue
        rows.append({"design_id": did, **sc})
    if not rows:
        sys.exit(f"[recompute] no folds found under {a.af3_dir}")
    folds = pd.DataFrame(rows).groupby("design_id", as_index=False).mean(
        numeric_only=True)
    print(f"[recompute] {len(folds)} designs scored from raw PAE matrices")

    df = load_dataset(a.dataset)
    merged = df.merge(folds, on="design_id", how="inner")
    print(f"[recompute] {len(merged)} designs joined to the benchmark")

    # --- E0 gate -------------------------------------------------------------
    if "ipsae_min_deposit" in merged:
        both = merged.dropna(subset=["ipsae_min_deposit", "ipsae_min"])
        rho = spearman(both.ipsae_min_deposit, both.ipsae_min)
        r = float(np.corrcoef(both.ipsae_min_deposit, both.ipsae_min)[0, 1])
        mad = float((both.ipsae_min_deposit - both.ipsae_min).abs().mean())
        print(f"[E0] vs deposit over {len(both)} designs: "
              f"Spearman {rho:.4f}, Pearson {r:.4f}, mean |diff| {mad:.4f}")
        worst = (both.assign(d=(both.ipsae_min_deposit - both.ipsae_min).abs())
                      .nlargest(5, "d")[["design_id", "target",
                                         "ipsae_min_deposit", "ipsae_min", "d"]])
        print("[E0] largest disagreements (section 5.3 failure cases):")
        print(worst.to_string(index=False))
        if a.gate and rho < a.gate_rho:
            sys.exit(f"[E0] FAILED: Spearman {rho:.4f} < {a.gate_rho}")
        if a.gate:
            print("[E0] PASSED")

    have = [c for c in AF3_FEATURES if c in merged.columns]
    missing = [c for c in AF3_FEATURES if c not in merged.columns]
    if missing:
        print(f"[recompute] NOTE {len(missing)} AF3 features unavailable in "
              f"these folds: {missing}")
    print(f"[recompute] {len(have)} AF3 features written")
    merged.to_csv(a.out, index=False)
    print(f"[recompute] -> {a.out}")


if __name__ == "__main__":
    main()
