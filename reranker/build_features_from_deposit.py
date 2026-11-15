#!/usr/bin/env python3
"""Build the re-ranker feature table from the published benchmark deposit.

This is the no-GPU path into Experiment 4.  It reads ``final_dataset.csv`` from
Zenodo record 15722219 (Overath et al., CC-BY-4.0) and emits the ``features.csv``
that ``train_reranker.py`` consumes.

Why this exists
---------------
``extract_features.py`` recomputes ipSAE from raw AF3 ``_confidences.json`` files
and therefore needs the 2 GB fold archive.  The deposit also ships *precomputed*
confidence scores in ``final_dataset.csv`` (82 MB), which is enough to train and
evaluate the re-ranker without touching a GPU or the archive at all.

The two paths do NOT give identical numbers: the deposit's ``af3_ipSAE_*``
columns come from the depositors' own implementation, which differs slightly
from ours (see ``--compare``).  Pick one and stay inside it -- never compare a
model trained here against a baseline measured by the other path.

Feature groups
--------------
``af3``    scores and descriptors derived from AF3's own prediction
``input``  descriptors of the designed complex, computed before any prediction

Other predictors in the deposit (AF2, ColabFold, Boltz-1) are deliberately
EXCLUDED: this is a re-ranker over AF3, and folding them in would answer a
different question than the one the report asks.

Usage
-----
    python build_features_from_deposit.py \
        --dataset /path/to/final_dataset.csv \
        --out features.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Targets dropped in the report: both pMHC screens, which contribute 3 of 394
# binders and are a different assay from the rest of the benchmark.
EXCLUDED_TARGETS = ["pMHC_SILSY1", "pMHC_NY1"]

# The 13 receptors the report keeps, for the gate check.
EXPECTED_TARGETS = [
    "IL10Ra", "Mdm2", "TrkA", "VirB8", "EGFR", "FGFR2", "InsulinR",
    "LTK", "Pdl1", "IL7Ra", "sntx", "IL2Ra", "SARS_CoV2_RBD",
]

GATES = {"n_designs": 3532, "n_positives": 391, "n_targets": 13}

# ---------------------------------------------------------------------------
# Column mapping: canonical feature name -> column in final_dataset.csv
# ---------------------------------------------------------------------------

IDENTITY = {
    "target": "target_id",
    "design": "binder_id",
}

# AF3 confidence scores.
AF3_SCORES = {
    "ipsae_min": "af3_ipSAE_min",
    "ipsae_max": "af3_ipSAE_max",
    "ipsae_avg": "af3_ipSAE_avg",
    "iptm": "af3_iptm_model_0",
    "ptm": "af3_ptm_model_0",
    "iptm_avg": "af3_iptm_avg",
    "iptm_std": "af3_iptm_std",
    "lis": "af3_LIS",
    "pdockq_min": "af3_pDockQ_min",
    "pdockq_max": "af3_pDockQ_max",
    "pdockq2_min": "af3_pDockQ2_min",
    "pdockq2_max": "af3_pDockQ2_max",
    "pae_interaction": "af3_ipae",
    "min_pae_contact": "af3_min_pae_contact",
    "max_pae_contact": "af3_max_pae_contact",
    "n_contact_res": "af3_res_above_contact_thres",
    "ipsae_d0chn": "af3_ipSAE_d0chn",
    "ipsae_n_in_calc": "af3_ipSAE_min_in_calculation",
}

# Geometry and energetics measured on AF3's PREDICTED structure.
AF3_STRUCTURE = {
    "af3_delta_sasa": "af3_pymol_delta_SASA",
    "af3_nonpolar_count": "af3_pymol_nonpolar_count",
    "af3_percent_nonpolar": "af3_pymol_percent_nonpolar",
    "af3_hbonds": "af3_pymol_hbonds",
    "af3_n_interface_res": "af3_pymol_num_interface_residues",
    "af3_interface_dG": "af3_rosetta_interface_dG",
    "af3_interface_sc": "af3_rosetta_interface_sc",
    "af3_interface_packstat": "af3_rosetta_interface_packstat",
    "af3_interface_dsasa": "af3_rosetta_interface_dSASA",
    "af3_dG_sasa_ratio": "af3_rosetta_interface_dG_SASA_ratio",
    "af3_unsat_hbonds": "af3_rosetta_interface_unsat_hbonds",
    "af3_sap_delta": "af3_sap_delta",
    "af3_binder_score": "af3_rosetta_binder_score",
    "rmsd_input_af3": "RMSD_complex_input_af3",
    "rmsd_chA_input_af3": "RMSD_chA_aft_chB_align_input_af3",
}

# Properties of the DESIGN itself, available before any structure prediction.
INPUT_FEATURES = {
    "binder_len": "A_length",
    "target_len": "B_length",
    "input_interface_dG": "input_rosetta_interface_dG",
    "input_interface_sc": "input_rosetta_interface_sc",
    "input_interface_packstat": "input_rosetta_interface_packstat",
    "input_interface_dsasa": "input_rosetta_interface_dSASA",
    "input_dG_sasa_ratio": "input_rosetta_interface_dG_SASA_ratio",
    "input_unsat_hbonds": "input_rosetta_interface_unsat_hbonds",
    "input_binder_score": "input_rosetta_binder_score",
    "input_surface_hydrophobicity": "input_rosetta_surface_hydrophobicity",
    "input_sap_delta": "input_sap_delta",
    "input_delta_sasa": "input_pymol_delta_SASA",
    "input_hbonds": "input_pymol_hbonds",
    "input_percent_nonpolar": "input_pymol_percent_nonpolar",
    "input_n_interface_res": "input_pymol_num_interface_residues",
    "camsol_score": "camsol_score",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(dataset: Path, keep_other_predictors: bool = False) -> pd.DataFrame:
    raw = pd.read_csv(dataset, low_memory=False)

    before = len(raw)
    raw = raw[raw["target_id"].notna() & raw["binder"].notna()].copy()
    after_blank = len(raw)
    raw = raw[~raw["target_id"].isin(EXCLUDED_TARGETS)].copy()
    after_pmhc = len(raw)

    print(f"rows in deposit                  : {before}")
    print(f"  after dropping incomplete rows : {after_blank}")
    print(f"  after dropping {len(EXCLUDED_TARGETS)} pMHC targets : {after_pmhc}")

    out = pd.DataFrame(index=raw.index)
    out["target"] = raw["target_id"].astype(str)
    out["design"] = raw["binder_id"].astype(str)
    out["label"] = raw["binder"].astype(bool).astype(int)

    missing = []
    groups = {"af3": {}, "input": {}}
    for canon, src in {**AF3_SCORES, **AF3_STRUCTURE}.items():
        if src in raw.columns:
            out[canon] = pd.to_numeric(raw[src], errors="coerce")
            groups["af3"][canon] = src
        else:
            missing.append(src)
    for canon, src in INPUT_FEATURES.items():
        if src in raw.columns:
            out[canon] = pd.to_numeric(raw[src], errors="coerce")
            groups["input"][canon] = src
        else:
            missing.append(src)

    if missing:
        print(f"\nWARNING: {len(missing)} expected columns absent: {missing}")

    print(f"\nfeatures built                   : {len(groups['af3'])} af3 + "
          f"{len(groups['input'])} input")

    # Median-impute within target, then globally, so a single missing cell does
    # not drop an entire design.
    feat_cols = list(groups["af3"]) + list(groups["input"])
    n_missing = int(out[feat_cols].isna().sum().sum())
    if n_missing:
        out[feat_cols] = (out.groupby("target")[feat_cols]
                             .transform(lambda s: s.fillna(s.median())))
        out[feat_cols] = out[feat_cols].fillna(out[feat_cols].median())
        print(f"imputed missing cells            : {n_missing} "
              f"({100 * n_missing / (len(out) * len(feat_cols)):.3f}%)")

    still_bad = [c for c in feat_cols if out[c].isna().any()]
    if still_bad:
        print(f"dropping all-NaN features        : {still_bad}")
        out = out.drop(columns=still_bad)
        for g in groups.values():
            for c in still_bad:
                g.pop(c, None)

    out.attrs["groups"] = groups
    return out


def check_gates(df: pd.DataFrame) -> bool:
    print("\n--- gates ---")
    ok = True
    checks = [
        ("designs", len(df), GATES["n_designs"]),
        ("positives", int(df["label"].sum()), GATES["n_positives"]),
        ("targets", df["target"].nunique(), GATES["n_targets"]),
    ]
    for name, got, want in checks:
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<10} {got:>6}  (expected {want})")

    unexpected = sorted(set(df["target"]) - set(EXPECTED_TARGETS))
    if unexpected:
        ok = False
        print(f"  FAIL  unexpected targets: {unexpected}")
    return bool(ok)


def compare_baselines(df: pd.DataFrame) -> None:
    """Print mean per-target AP for each raw score, for provenance."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from metrics import average_precision, per_target_scores

    y = df["label"].values
    tg = df["target"].values
    print("\n--- baseline scores on this table ---")
    for col in ["ipsae_min", "ipsae_max", "ipsae_avg", "iptm", "lis"]:
        if col not in df.columns:
            continue
        pt = per_target_scores(tg, y, df[col].values, min_positives=1,
                               n_tiebreaks=200, seed=0)
        aps = [v["ap"] for v in pt.values()]
        pooled = average_precision(y, df[col].values, n_tiebreaks=200)
        print(f"  {col:<12} mean per-target AP = {np.mean(aps):.4f}   "
              f"pooled AP = {pooled:.4f}")
    print(f"  {'random floor':<12} {y.mean():.4f}")
    print("\n  NOTE: the report's 0.517 comes from ipSAE recomputed by "
          "score_e2.py from\n  raw PAE matrices, not from these precomputed "
          "columns. The two differ.\n  Use ONE source for both the model and "
          "its baseline.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="final_dataset.csv from Zenodo record 15722219")
    ap.add_argument("--out", default=Path("features.csv"), type=Path)
    ap.add_argument("--no-gates", action="store_true",
                    help="write the table even if the gates fail")
    ap.add_argument("--compare", action="store_true",
                    help="also print baseline AP for each raw score")
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"error: {args.dataset} not found", file=sys.stderr)
        return 1

    df = build(args.dataset)
    passed = check_gates(df)

    if not passed and not args.no_gates:
        print("\nGates failed. The deposit may have been revised, or the "
              "exclusion rule has changed.\nInvestigate before using this "
              "table; --no-gates overrides, but a table that fails the\ngates "
              "will not reproduce the numbers already in the report.",
              file=sys.stderr)
        return 1

    if args.compare:
        compare_baselines(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows x {len(df.columns)} cols)")

    prov = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": str(args.dataset),
        "source_sha256": sha256(args.dataset),
        "zenodo_record": "15722219",
        "zenodo_doi": "10.5281/zenodo.15722219",
        "license": "CC-BY-4.0",
        "excluded_targets": EXCLUDED_TARGETS,
        "n_designs": len(df),
        "n_positives": int(df["label"].sum()),
        "n_targets": int(df["target"].nunique()),
        "gates_passed": passed,
        "feature_groups": df.attrs.get("groups", {}),
        "note": ("ipSAE here is the deposit's precomputed af3_ipSAE_* column, "
                 "not recomputed from raw PAE by score_e2.py; the two differ "
                 "slightly."),
    }
    side = args.out.with_suffix(".provenance.json")
    side.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
