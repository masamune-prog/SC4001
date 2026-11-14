"""Feature table from the deposit's published scores.

The 16 design-side features are properties of the designed complex computed
*before* any structure prediction: binder and target length, Rosetta interface
terms, aggregation propensity, and PyMOL interface descriptors (section 3.5).
They come straight out of the deposit's CSV; this script resolves their column
names, z-scores them within each target, and writes the table that the Study B
re-ranker consumes alongside the AF3 outputs.

Within-target z-scoring is a modelling choice, not a neutral one: it removes
the between-campaign offsets that a target-blind model would otherwise latch
onto.  The pooled alternative is one of the design ablations.

NOTE: the re-ranker itself (train_reranker.py / eval_reranker.py) is not part
of this reconstruction.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from dataset import carry_through_columns, load_dataset

# The 16 design-side features, with the aliases the deposit is likely to use.
INPUT_FEATURES = {
    "binder_len":        ["binder_len", "binder_length", "length_binder"],
    "target_len":        ["target_len", "target_length", "length_target"],
    "rosetta_dG":        ["dg", "ddg", "interface_dg", "rosetta_dg", "binding_energy"],
    "shape_complementarity": ["shape_complementarity", "sc", "interface_sc"],
    "packstat":          ["packstat", "interface_packstat"],
    "dSASA":             ["dsasa", "interface_dsasa", "delta_sasa"],
    "dG_over_SASA":      ["dg_sasa", "dg_per_sasa", "interface_dg_sasa"],
    "unsat_hbonds":      ["unsat", "unsathbonds", "interface_hbond_unsat",
                          "delta_unsat_hbonds"],
    "binder_score":      ["binder_score", "monomer_score"],
    "surface_hydrophobicity": ["surface_hydrophobicity", "hydrophobicity"],
    "sap":               ["sap", "sap_score", "aggregation_sap"],
    "camsol":            ["camsol", "camsol_score", "solubility"],
    "n_interface_res":   ["n_interface_res", "interface_residues", "n_contacts"],
    "interface_area":    ["interface_area", "buried_area", "bsa"],
    "interface_hbonds":  ["interface_hbonds", "hbonds", "n_hbonds"],
    "interface_polar_frac": ["interface_polar", "polar_fraction", "frac_polar"],
}


def resolve_feature(raw: pd.DataFrame, aliases: list[str]) -> str | None:
    lowered = {c.lower().replace(" ", "_"): c for c in raw.columns}
    for al in aliases:
        if al in lowered:
            return lowered[al]
    for al in aliases:
        for low, orig in lowered.items():
            if al in low:
                return orig
    return None


def zscore_within(df: pd.DataFrame, cols: list[str], group: str = "target",
                  pooled: bool = False) -> pd.DataFrame:
    out = df.copy()
    if pooled:
        for c in cols:
            s = out[c].astype(float)
            out[c] = (s - s.mean()) / (s.std(ddof=0) or 1.0)
        return out
    for c in cols:
        out[c] = (out.groupby(group)[c]
                     .transform(lambda s: (s - s.mean()) /
                                (s.std(ddof=0) if s.std(ddof=0) else 1.0)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--af3-features", default=None,
                   help="features_folds.csv from recompute_ipsae_from_folds.py")
    p.add_argument("--out", default="features_input.csv")
    p.add_argument("--pooled", action="store_true",
                   help="z-score pooled across targets instead of within")
    p.add_argument("--no-zscore", action="store_true")
    a = p.parse_args()

    base = load_dataset(a.dataset)
    raw = carry_through_columns(a.dataset)
    key = None
    for cand in ("design_id", "design", "name", "id"):
        hit = [c for c in raw.columns if c.lower() == cand]
        if hit:
            key = hit[0]
            break
    if key is None:
        sys.exit("[features] cannot find a design-id column in the deposit")
    raw = raw.assign(design_id=raw[key].astype(str).str.strip())

    cols, missing = {}, []
    for name, aliases in INPUT_FEATURES.items():
        src = resolve_feature(raw.drop(columns=["design_id"]), aliases)
        if src is None:
            missing.append(name)
        else:
            cols[name] = src

    feat = pd.DataFrame({"design_id": raw.design_id})
    for name, src in cols.items():
        feat[name] = pd.to_numeric(raw[src], errors="coerce")

    df = base.merge(feat, on="design_id", how="left")
    # lengths are always derivable, whatever the deposit calls them
    df["binder_len"] = df.binder_seq.str.len()
    df["target_len"] = df.target_seq.str.len()
    missing = [m for m in missing if m not in ("binder_len", "target_len")]

    feature_cols = [c for c in INPUT_FEATURES if c in df.columns]
    print(f"[features] resolved {len(feature_cols)} of {len(INPUT_FEATURES)} "
          f"design-side features")
    if missing:
        print(f"[features] MISSING: {missing}")
        print("[features] check the deposit's column names and extend "
              "INPUT_FEATURES before trusting the counts in section 3.5")

    # median-impute, then z-score (within target by default)
    for c in feature_cols:
        df[c] = df[c].fillna(df.groupby("target")[c].transform("median"))
        df[c] = df[c].fillna(df[c].median())
    if not a.no_zscore:
        df = zscore_within(df, feature_cols, pooled=a.pooled)

    if a.af3_features:
        af3 = pd.read_csv(a.af3_features)
        drop = [c for c in af3.columns if c in df.columns and c != "design_id"]
        df = df.merge(af3.drop(columns=drop), on="design_id", how="left")
        print(f"[features] joined {af3.shape[1] - 1} AF3 columns")

    df = df.drop(columns=["binder_seq", "target_seq"], errors="ignore")
    df.to_csv(a.out, index=False)
    n_af3 = sum(1 for c in df.columns if c.startswith(("ipsae", "ipae", "plddt",
                                                       "contact_prob", "chain_pair"))
                or c in ("iptm", "ptm", "ranking_score", "fraction_disordered"))
    print(f"[features] {len(df)} rows, {len(feature_cols)} design-side + "
          f"{n_af3} AF3 columns -> {a.out}")


if __name__ == "__main__":
    main()
