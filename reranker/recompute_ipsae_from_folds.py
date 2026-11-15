#!/usr/bin/env python3
"""Recompute ipSAE from the deposit's raw AF3 folds, then build the feature table.

This is the high-fidelity path into Experiment 4.  ``build_features_from_deposit.py``
reads the depositors' *precomputed* ``af3_ipSAE_*`` columns, which do not match
the report's own numbers because they came from a different implementation.  This
script instead recomputes ipSAE from each fold's raw PAE matrix using
``extract_features.ipsae_from_pae`` -- the same equation the report's Baseline
section states -- so the re-ranker's baseline lines up with the rest of the paper.

Inputs
------
``--af3-dir``   directory holding ``AF3_outputs/<design>/<design>_confidences.json``
                (extract from AF3_outputs.tar.zstd, Zenodo record 15722219)
``--dataset``   ``final_dataset.csv`` from the same deposit, for labels, targets
                and the design-side input features

Output
------
``features_folds.csv``, in the same schema ``train_reranker.py`` already consumes,
with ``ipsae_min`` / ``ipsae_max`` recomputed rather than copied.

Usage
-----
    python recompute_ipsae_from_folds.py \
        --af3-dir /path/to/extracted \
        --dataset /path/to/final_dataset.csv \
        --out features_folds.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import ipsae_from_pae  # noqa: E402
from build_features_from_deposit import (  # noqa: E402
    EXCLUDED_TARGETS, EXPECTED_TARGETS, GATES, INPUT_FEATURES, sha256,
)

BINDER_CHAIN = "A"


def fold_features(design_dir: Path, design: str) -> dict | None:
    """Everything derivable from one design's two confidence JSONs."""
    full = design_dir / f"{design.lower()}_confidences.json"
    summ = design_dir / f"{design.lower()}_summary_confidences.json"
    if not full.exists():
        cand = sorted(p for p in design_dir.glob("*_confidences.json")
                      if "summary" not in p.name)
        if not cand:
            return None
        full = cand[0]
    if not summ.exists():
        cand = sorted(design_dir.glob("*_summary_confidences.json"))
        summ = cand[0] if cand else None

    try:
        with open(full, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    out: dict[str, float] = {}
    chain_ids = data.get("token_chain_ids")
    pae = data.get("pae")
    if pae is None or not chain_ids:
        return None

    pae_arr = np.asarray(pae, dtype=np.float64)
    arr = np.asarray(chain_ids)
    chains = sorted(set(arr.tolist()))

    # The report's ipSAE is defined for a two-chain complex. Designs with extra
    # target subchains are not part of the 13-receptor set and are dropped
    # upstream, but guard anyway rather than silently producing a wrong number.
    if len(chains) != 2:
        return None

    lo, hi = ipsae_from_pae(pae_arr, list(chain_ids))
    out["ipsae_min"] = lo
    out["ipsae_max"] = hi
    out["ipsae_avg"] = 0.5 * (lo + hi)

    a = np.where(arr == chains[0])[0]
    b = np.where(arr == chains[1])[0]
    cross = pae_arr[np.ix_(a, b)]
    out["pae_interface_mean"] = float(cross.mean())
    out["pae_interface_min"] = float(cross.min())
    out["pae_interface_frac_below_10"] = float((cross < 10).mean())
    out["n_interface_pairs"] = float((cross < 10).sum())
    out["pae_interaction"] = float(cross.mean())

    cp = data.get("contact_probs")
    if cp is not None:
        cpa = np.asarray(cp, dtype=np.float64)[np.ix_(a, b)]
        out["contact_prob_max"] = float(cpa.max())
        out["contact_prob_sum"] = float(cpa.sum())

    plddts = data.get("atom_plddts")
    atom_chains = data.get("atom_chain_ids")
    if plddts:
        p = np.asarray(plddts, dtype=np.float64)
        out["plddt_mean"] = float(p.mean())
        out["plddt_min"] = float(p.min())
        out["plddt_frac_above_70"] = float((p > 70).mean())
        if atom_chains and len(atom_chains) == len(p):
            ac = np.asarray(atom_chains)
            mb = ac == BINDER_CHAIN
            if mb.any():
                out["plddt_binder_mean"] = float(p[mb].mean())
            if (~mb).any():
                out["plddt_target_mean"] = float(p[~mb].mean())

    if summ is not None and summ.exists():
        try:
            with open(summ, "r", encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, json.JSONDecodeError):
            s = {}
        for k in ("iptm", "ptm", "ranking_score", "fraction_disordered",
                  "has_clash"):
            if isinstance(s.get(k), (int, float)):
                out[k if k in ("iptm", "ptm") else f"af3_{k}"] = float(s[k])
        cpi = s.get("chain_pair_iptm")
        if cpi:
            m = np.asarray(cpi, dtype=np.float64)
            if m.shape[0] >= 2:
                out["chain_pair_iptm_ab"] = float(m[0, 1])
        cpp = s.get("chain_pair_pae_min")
        if cpp:
            m = np.asarray(cpp, dtype=np.float64)
            if m.shape[0] >= 2:
                out["chain_pair_pae_min_ab"] = float(m[0, 1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--af3-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", default=Path("features_folds.csv"), type=Path)
    ap.add_argument("--no-gates", action="store_true")
    args = ap.parse_args()

    raw = pd.read_csv(args.dataset, low_memory=False)
    raw = raw[raw["target_id"].notna() & raw["binder"].notna()]
    raw = raw[~raw["target_id"].isin(EXCLUDED_TARGETS)].copy()
    print(f"designs to process : {len(raw)}")

    root = args.af3_dir
    if (root / "AF3_outputs").is_dir():
        root = root / "AF3_outputs"

    rows, missing, skipped = [], [], []
    t0 = time.time()
    total = len(raw)
    for i, (_, r) in enumerate(raw.iterrows(), 1):
        design = str(r["binder_id"])
        ddir = root / design
        if not ddir.is_dir():
            missing.append(design)
            continue
        feats = fold_features(ddir, design)
        if feats is None:
            skipped.append(design)
            continue
        feats["design"] = design
        feats["target"] = str(r["target_id"])
        feats["label"] = int(bool(r["binder"]))
        rows.append(feats)
        if i % 250 == 0:
            el = time.time() - t0
            print(f"  {i}/{total}  ({el:.0f}s elapsed, "
                  f"~{el / i * (total - i):.0f}s left)", flush=True)

    print(f"parsed             : {len(rows)}")
    if missing:
        print(f"no fold directory  : {len(missing)}  e.g. {missing[:3]}")
    if skipped:
        print(f"unusable JSON      : {len(skipped)}  e.g. {skipped[:3]}")

    df = pd.DataFrame(rows)

    # Attach the design-side input features from the deposit table.
    keep = {src: canon for canon, src in INPUT_FEATURES.items()
            if src in raw.columns}
    side = raw[["binder_id", *keep]].rename(columns={"binder_id": "design", **keep})
    df = df.merge(side, on="design", how="left")

    front = ["target", "design", "label"]
    df = df[front + [c for c in df.columns if c not in front]]

    feat_cols = [c for c in df.columns if c not in front]
    n_missing = int(df[feat_cols].isna().sum().sum())
    if n_missing:
        df[feat_cols] = (df.groupby("target")[feat_cols]
                           .transform(lambda s: s.fillna(s.median())))
        df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median())
        print(f"imputed cells      : {n_missing}")
    dead = [c for c in feat_cols if df[c].isna().all()]
    if dead:
        print(f"dropping all-NaN   : {dead}")
        df = df.drop(columns=dead)

    print("\n--- gates ---")
    ok = True
    for name, got, want in [("designs", len(df), GATES["n_designs"]),
                            ("positives", int(df["label"].sum()), GATES["n_positives"]),
                            ("targets", df["target"].nunique(), GATES["n_targets"])]:
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<10} {got:>6}  (expected {want})")
    bad = sorted(set(df["target"]) - set(EXPECTED_TARGETS))
    if bad:
        ok = False
        print(f"  FAIL  unexpected targets: {bad}")

    from metrics import average_precision, per_target_scores
    y = df["label"].values
    tg = df["target"].values
    print("\n--- recomputed baselines ---")
    for col in ["ipsae_min", "ipsae_max", "ipsae_avg", "iptm"]:
        if col not in df.columns:
            continue
        pt = per_target_scores(tg, y, df[col].values, min_positives=1,
                               n_tiebreaks=200, seed=0)
        aps = [v["ap"] for v in pt.values()]
        pooled = average_precision(y, df[col].values, n_tiebreaks=200)
        print(f"  {col:<12} mean per-target AP = {np.mean(aps):.4f}   "
              f"pooled AP = {pooled:.4f}")
    print("  report quotes: mean per-target AP 0.517, pooled AP 0.350")

    if not ok and not args.no_gates:
        print("\nGates failed; not writing.", file=sys.stderr)
        return 1

    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows x {len(df.columns)} cols)")

    prov = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "af3_dir": str(args.af3_dir),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "zenodo_record": "15722219",
        "license": "CC-BY-4.0",
        "ipsae": "recomputed from raw PAE via extract_features.ipsae_from_pae "
                 "(10 A cutoff, report's d0)",
        "n_designs": len(df),
        "n_positives": int(df["label"].sum()),
        "gates_passed": bool(ok),
    }
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
