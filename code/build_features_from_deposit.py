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

from dataset import carry_through_columns, load_dataset, resolve

# Prefixes marking a column computed FROM a predicted structure.  A design-side
# feature is by definition available before any structure prediction, so a
# feature resolving into one of these is not a naming quirk -- it is a leak of
# prediction-derived information into the "input" block, and it silently
# inflates anything trained on it.  The deposit carries an af2_/af3_/colab_/
# boltz1_ twin of almost every input_ column, so the loose fallback below will
# hit one unless it is told not to.
PREDICTION_PREFIXES = ("af2_", "af3_", "colab_", "boltz1_")

# The 16 design-side features.  The deposit's own column name is listed FIRST so
# the exact-match pass wins before the loose contains-match is ever reached.
#
# Two of these were silently mis-resolving before the deposit's names were
# pinned here, and both were worse than an outright miss:
#   * shape_complementarity matched "camsol_score", because the two-letter
#     alias "sc" is a substring of "_score" -- so shape complementarity and
#     camsol were the same column twice over.  It is input_rosetta_interface_sc.
#   * n_interface_res matched "af2_pymol_num_interface_residues", an
#     AlphaFold2-derived count, i.e. exactly the leak described above.
INPUT_FEATURES = {
    "binder_len":        ["binder_len", "binder_length", "length_binder"],
    "target_len":        ["target_len", "target_length", "length_target"],
    "rosetta_dG":        ["input_rosetta_interface_dg", "interface_dg",
                          "rosetta_dg", "binding_energy", "dg"],
    "shape_complementarity": ["input_rosetta_interface_sc",
                              "shape_complementarity", "interface_sc"],
    "packstat":          ["input_rosetta_interface_packstat", "packstat",
                          "interface_packstat"],
    "dSASA":             ["input_rosetta_interface_dsasa", "dsasa",
                          "interface_dsasa", "delta_sasa"],
    "dG_over_SASA":      ["input_rosetta_interface_dg_sasa_ratio", "dg_sasa",
                          "dg_per_sasa", "interface_dg_sasa"],
    "unsat_hbonds":      ["input_rosetta_interface_unsat_hbonds", "unsat",
                          "unsathbonds", "interface_hbond_unsat",
                          "delta_unsat_hbonds"],
    "binder_score":      ["input_rosetta_binder_score", "binder_score",
                          "monomer_score"],
    "surface_hydrophobicity": ["input_rosetta_surface_hydrophobicity",
                               "surface_hydrophobicity", "hydrophobicity"],
    "sap":               ["input_sap_binder", "sap", "sap_score",
                          "aggregation_sap"],
    "camsol":            ["camsol_score", "camsol", "solubility"],
    "n_interface_res":   ["input_pymol_num_interface_residues",
                          "n_interface_res", "interface_residues", "n_contacts"],
    "interface_area":    ["input_pymol_delta_sasa", "interface_area",
                          "buried_area", "bsa"],
    "interface_hbonds":  ["input_rosetta_interface_interface_hbonds",
                          "input_pymol_hbonds", "interface_hbonds", "hbonds",
                          "n_hbonds"],
    # The deposit stores the NONpolar percentage; the polar fraction is its
    # complement, derived in main() rather than read off a column.
    "interface_polar_frac": ["input_pymol_percent_nonpolar"],
}

# Features whose stored column is the complement of the feature we want, as a
# percentage: value = 100 - column.
COMPLEMENT_OF_PERCENT = {"interface_polar_frac"}


def resolve_feature(raw: pd.DataFrame, aliases: list[str],
                    allow_prediction_derived: bool = False) -> str | None:
    """Resolve one design-side feature to a column of the deposit.

    Columns computed from a predicted structure are refused unless explicitly
    allowed, so the loose contains-match cannot wander into an af2_/af3_ twin
    of the intended input_ column.
    """
    lowered = {c.lower().replace(" ", "_"): c for c in raw.columns}
    if not allow_prediction_derived:
        lowered = {low: orig for low, orig in lowered.items()
                   if not low.startswith(PREDICTION_PREFIXES)}
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
    # Resolve the design-id column through dataset.ALIASES rather than a second,
    # private list.  The private list only matched a column named exactly
    # "design_id"/"design"/"name"/"id", so it failed outright on the deposit,
    # whose column is "binder_id" -- and had it matched something else, the
    # join below would have silently produced all-NaN features.  One alias
    # table, one place to fix.
    try:
        key = resolve(raw, "design_id")
    except KeyError as exc:
        sys.exit(f"[features] cannot find a design-id column in the deposit:\n{exc}")
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
        v = pd.to_numeric(raw[src], errors="coerce")
        if name in COMPLEMENT_OF_PERCENT:
            # stored as the nonpolar percentage; we want the polar fraction
            v = 100.0 - v
        feat[name] = v

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
