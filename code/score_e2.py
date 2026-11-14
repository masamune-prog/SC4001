"""Score every completed fold -> e2_scores.tsv.

Walks one or more AF3 output roots, recomputes ipSAE from each fold's PAE
matrix (coordinates are never read), attaches the design's condition, seed and
experimental label, and writes one long table that every analysis consumes.

Folds that have not finished are skipped and counted, so this can be run
against a partially complete sweep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from af3_io import find_confidences, parse_fold_name
from dataset import load_dataset
from ipsae_recompute import PAE_CUTOFF, score_fold


def scan(roots: list[str], chain1: str, chain2: str, cutoff: float) -> pd.DataFrame:
    rows, skipped, failed = [], 0, 0
    for root in roots:
        pairs = find_confidences(root)
        print(f"[score_e2] {root}: {len(pairs)} completed folds", file=sys.stderr)
        for conf, summary in pairs:
            meta = parse_fold_name(conf.parent.name)
            if meta is None:
                # tolerate AF3's own subdirectory layout (seed-*/sample-*)
                meta = parse_fold_name(conf.parent.parent.name)
            if meta is None:
                skipped += 1
                continue
            try:
                sc = score_fold(conf, summary, chain1, chain2, cutoff)
            except Exception as exc:                       # noqa: BLE001
                print(f"[score_e2] FAILED {conf}: {exc}", file=sys.stderr)
                failed += 1
                continue
            rows.append({**meta, **sc, "fold_dir": str(conf.parent)})
    print(f"[score_e2] scored {len(rows)}, unparseable {skipped}, failed {failed}",
          file=sys.stderr)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+",
                   default=["e2_out_full", "e2_out_none"],
                   help="AF3 output directories to scan")
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--out", default="e2_scores.tsv")
    p.add_argument("--chain1", default="A", help="target chain")
    p.add_argument("--chain2", default="B", help="binder chain")
    p.add_argument("--cutoff", type=float, default=PAE_CUTOFF)
    a = p.parse_args()

    scores = scan(a.roots, a.chain1, a.chain2, a.cutoff)
    if scores.empty:
        sys.exit("[score_e2] nothing scored")

    df = load_dataset(a.dataset)[["design_id", "target", "target_uid", "label",
                                  "binder_seq", "target_seq"]]
    df["binder_len"] = df.binder_seq.str.len()
    df["target_len"] = df.target_seq.str.len()
    merged = scores.merge(df.drop(columns=["binder_seq", "target_seq"]),
                          on="design_id", how="left")
    unmatched = merged.target.isna().sum()
    if unmatched:
        print(f"[score_e2] WARNING {unmatched} folds have no dataset row "
              f"(design id mismatch?)", file=sys.stderr)

    merged.to_csv(a.out, sep="\t", index=False)
    print(f"[score_e2] -> {a.out}  ({len(merged)} rows)")
    print(merged.groupby(["condition", "seed"]).agg(
        n=("ipsae_min", "size"),
        zero_frac=("ipsae_min", lambda s: float((s == 0).mean())),
        mean_ipsae_min=("ipsae_min", "mean")))


if __name__ == "__main__":
    main()
