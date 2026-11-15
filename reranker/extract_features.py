"""
Build the re-ranker feature table from the Overath et al. benchmark deposit.

    Zenodo record 15722219 (CC-BY-4.0)
      final_dataset.csv     82 MB   labels, sequences, deposited scores
      AF3_outputs.tar.zstd  2.0 GB  AF3 confidence JSONs (optional, richer feats)

The script reproduces the filtering described in the report's Dataset section
and then refuses to continue unless the result matches the numbers already
published there.  Those five gates are the whole point of this file: an MLP
evaluated on a slightly different row set cannot be compared against the
ablation results, and the mismatch would be invisible in the output.

  Gate 1   3,532 designs after filtering
  Gate 2   391 positives (10.7%)
  Gate 3   13 distinct targets
  Gate 4   mean per-target AP on ipSAE_min = 0.517
  Gate 5   pooled AP on ipSAE_min = 0.350

Gates 1-3 are structural and must pass exactly.  Gates 4-5 are checked to two
decimals, since tie-breaking makes the third decimal stochastic.

Usage
-----
    # 1. see what columns the deposit actually ships
    python extract_features.py --data-dir ../data --inspect

    # 2. build the table (writes features.csv + a provenance sidecar)
    python extract_features.py --data-dir ../data --out ../data/features.csv

    # 3. optionally fold in AF3 JSON-derived features
    python extract_features.py --data-dir ../data --out ../data/features.csv \
        --af3-dir ../data/AF3_outputs

If a gate fails, the message tells you which filter to look at.  Do not pass
--no-gates to make a failure go away; ask the teammate who ran the folds how
they got from 3,669 rows to 3,532.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import average_precision, per_target_scores

# --------------------------------------------------------------------------
# What the report says the data should look like
# --------------------------------------------------------------------------

# The thirteen kept targets, read off Figure 1 of the report.  Anything in the
# deposit that is not on this list is one of the two excluded pMHC targets.
EXPECTED_TARGETS = [
    "IL10Ra", "Mdm2", "TrkA", "VirB8", "EGFR", "FGFR2", "InsulinR",
    "LTK", "Pdl1", "IL7Ra", "sntx", "IL2Ra", "SARS_CoV2_RBD",
]

GATES = {
    "n_designs": 3532,
    "n_positives": 391,
    "n_targets": 13,
    "mean_per_target_ap": 0.517,
    "pooled_ap": 0.350,
}

# Column name candidates, tried in order.  The deposit's exact header is not
# documented in the report, so we auto-detect and print what we found.  Use
# --column-map to override with a JSON file if auto-detection picks wrong.
CANDIDATES = {
    "target": ["target", "target_name", "receptor", "target_id", "Target"],
    "design": ["binder_name", "design", "design_id", "name", "binder_id", "id"],
    "label": ["binds", "label", "binder", "is_binder", "success", "hit",
              "experimental_result", "binding"],
    "ipsae_min": ["ipsae_min", "ipSAE_min", "ipsae_d0chn_min", "ipsae"],
    "ipsae_max": ["ipsae_max", "ipSAE_max", "ipsae_d0chn_max"],
    "iptm": ["iptm", "ipTM", "af3_iptm"],
    "ptm": ["ptm", "pTM", "af3_ptm"],
    "plddt": ["plddt", "pLDDT", "mean_plddt", "af3_plddt"],
    "pae_interaction": ["pae_interaction", "pae_int", "mean_pae_interface",
                        "interface_pae"],
    "binder_seq": ["binder_sequence", "binder_seq", "sequence", "peptide_sequence"],
    "target_seq": ["target_sequence", "target_seq", "receptor_sequence"],
    "binder_len": ["binder_length", "binder_len", "length"],
    "target_len": ["target_length", "target_len"],
}

# Values that count as a confirmed binder, lower-cased.
TRUTHY = {"1", "1.0", "true", "yes", "y", "binder", "bind", "binds",
          "positive", "success", "t"}


# --------------------------------------------------------------------------
# Schema discovery
# --------------------------------------------------------------------------

def normalise(name: str) -> str:
    """Loose key for matching target names across spelling conventions."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def detect_columns(df: pd.DataFrame, override: dict | None = None) -> dict:
    """Map our canonical field names onto the deposit's actual columns."""
    override = override or {}
    lowered = {c.lower(): c for c in df.columns}
    found: dict[str, str | None] = {}

    for field, options in CANDIDATES.items():
        if field in override:
            col = override[field]
            if col not in df.columns:
                raise SystemExit(
                    f"--column-map says {field!r} -> {col!r} but that column "
                    f"is not in the file."
                )
            found[field] = col
            continue
        found[field] = next(
            (lowered[o.lower()] for o in options if o.lower() in lowered), None
        )
    return found


def inspect(path: Path) -> None:
    """Print the deposit's schema so the column map can be filled in by hand."""
    df = pd.read_csv(path, nrows=2000)
    print(f"\n{path.name}: {len(df.columns)} columns "
          f"(showing dtypes from the first 2,000 rows)\n")
    for col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        print(f"  {col:<36} {str(df[col].dtype):<10} e.g. {sample}")

    print("\nAuto-detected mapping:")
    for field, col in detect_columns(df).items():
        mark = "ok " if col else "-- "
        print(f"  {mark} {field:<18} -> {col}")
    print("\nAnything marked '--' needs a --column-map entry, or is optional.\n")


# --------------------------------------------------------------------------
# AF3 confidence JSON parsing (optional, richer features)
# --------------------------------------------------------------------------

def d0(n: float) -> float:
    """ipSAE distance scale, exactly as written in the report's Baseline."""
    return max(1.0, 1.24 * (max(n, 26.0) - 15.0) ** (1.0 / 3.0) - 1.8)


def ipsae_from_pae(pae: np.ndarray, chain_ids: list[str], cutoff: float = 10.0
                   ) -> tuple[float, float]:
    """Recompute ipSAE in both chain directions from an AF3 PAE matrix.

    Mirrors the equation in the report: for each residue i in chain c1, average
    the PAE-derived term over its valid partners j in c2 (those with
    PAE(i,j) < cutoff), then take the maximum over i.  Returns (min, max) of
    the two directions.
    """
    chains = sorted(set(chain_ids))
    if len(chains) != 2:
        return float("nan"), float("nan")

    arr = np.asarray(chain_ids)
    idx = {c: np.where(arr == c)[0] for c in chains}
    values = []
    for c1, c2 in [(chains[0], chains[1]), (chains[1], chains[0])]:
        block = pae[np.ix_(idx[c1], idx[c2])]
        valid = block < cutoff
        n0 = valid.sum(axis=1)
        best = 0.0
        for i in range(block.shape[0]):
            if n0[i] == 0:
                continue
            scale = d0(float(n0[i]))
            term = 1.0 / (1.0 + (block[i][valid[i]] / scale) ** 2)
            best = max(best, float(term.sum() / n0[i]))
        values.append(best)
    return float(min(values)), float(max(values))


def features_from_af3_json(path: Path) -> dict | None:
    """Pull summary features out of one AF3 *_confidences.json file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    out: dict[str, float] = {}
    for key in ("iptm", "ptm", "fraction_disordered", "ranking_score"):
        if key in data and isinstance(data[key], (int, float)):
            out[f"af3_{key}"] = float(data[key])

    plddts = data.get("atom_plddts") or data.get("plddt")
    if plddts:
        p = np.asarray(plddts, dtype=np.float64)
        out["plddt_mean"] = float(p.mean())
        out["plddt_min"] = float(p.min())
        out["plddt_frac_above_70"] = float((p > 70).mean())

    pae = data.get("pae")
    chain_ids = data.get("token_chain_ids")
    if pae is not None and chain_ids:
        pae_arr = np.asarray(pae, dtype=np.float64)
        lo, hi = ipsae_from_pae(pae_arr, list(chain_ids))
        out["ipsae_min_recomputed"] = lo
        out["ipsae_max_recomputed"] = hi

        arr = np.asarray(chain_ids)
        chains = sorted(set(arr.tolist()))
        if len(chains) == 2:
            a = np.where(arr == chains[0])[0]
            b = np.where(arr == chains[1])[0]
            cross = pae_arr[np.ix_(a, b)]
            out["pae_interface_mean"] = float(cross.mean())
            out["pae_interface_min"] = float(cross.min())
            out["pae_interface_frac_below_10"] = float((cross < 10).mean())
            out["n_interface_pairs"] = float((cross < 10).sum())
    return out


def collect_af3_features(af3_dir: Path) -> pd.DataFrame:
    """Walk a directory of AF3 outputs and build a per-design feature frame."""
    rows = []
    files = sorted(af3_dir.rglob("*confidences*.json"))
    if not files:
        print(f"  warning: no *confidences*.json under {af3_dir}", file=sys.stderr)
    for i, path in enumerate(files, 1):
        feats = features_from_af3_json(path)
        if feats is None:
            continue
        # Design id is taken from the containing directory, which is how AF3
        # names its per-fold output folders.
        feats["design"] = path.parent.name
        rows.append(feats)
        if i % 500 == 0:
            print(f"  parsed {i}/{len(files)} JSONs", file=sys.stderr)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def coerce_label(series: pd.Series) -> np.ndarray:
    """Turn whatever the deposit uses for the outcome into 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) > 0.5).astype(int).to_numpy()
    return (series.astype(str).str.strip().str.lower().isin(TRUTHY)
            .astype(int).to_numpy())


def build(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    csv_path = data_dir / args.csv_name
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found.  Download it first:\n"
            f"  curl -L -o {csv_path} \\\n"
            f"    https://zenodo.org/api/records/15722219/files/"
            f"final_dataset.csv/content"
        )

    override = {}
    if args.column_map:
        override = json.loads(Path(args.column_map).read_text(encoding="utf-8"))

    print(f"reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    cols = detect_columns(df, override)
    required = ["target", "label", "ipsae_min"]
    missing = [f for f in required if cols.get(f) is None]
    if missing:
        raise SystemExit(
            f"could not auto-detect required column(s): {missing}\n"
            f"Run with --inspect to see the header, then supply a JSON file "
            f"via --column-map, e.g. {{\"label\": \"binds_yn\"}}"
        )
    print("  column mapping:")
    for field, col in cols.items():
        if col:
            print(f"    {field:<18} -> {col}")

    out = pd.DataFrame()
    out["target_raw"] = df[cols["target"]].astype(str)
    out["design"] = (df[cols["design"]].astype(str) if cols.get("design")
                     else [f"design_{i}" for i in range(len(df))])
    out["label"] = coerce_label(df[cols["label"]])

    # ---- numeric features straight from the CSV --------------------------
    for field in ["ipsae_min", "ipsae_max", "iptm", "ptm", "plddt",
                  "pae_interaction", "binder_len", "target_len"]:
        col = cols.get(field)
        if col is not None:
            out[field] = pd.to_numeric(df[col], errors="coerce")

    # Lengths derived from sequences when not shipped as columns.
    if "binder_len" not in out and cols.get("binder_seq"):
        out["binder_len"] = df[cols["binder_seq"]].astype(str).str.len()
    if "target_len" not in out and cols.get("target_seq"):
        out["target_len"] = df[cols["target_seq"]].astype(str).str.len()

    # ---- filter 1: designs with a usable score ---------------------------
    before = len(out)
    out = out[out["ipsae_min"].notna()].copy()
    print(f"\nfilter: dropped {before - len(out):,} rows without an ipSAE score "
          f"-> {len(out):,} remain")

    # ---- filter 2: the two pMHC targets ----------------------------------
    keep_keys = {normalise(t) for t in EXPECTED_TARGETS}
    out["target"] = out["target_raw"]
    seen = {normalise(t): t for t in out["target_raw"].unique()}
    dropped = [name for key, name in seen.items() if key not in keep_keys]
    if dropped:
        print(f"filter: excluding {len(dropped)} target(s) not on the kept list "
              f"(the pMHC exclusion): {dropped}")
    out = out[out["target_raw"].map(lambda t: normalise(t) in keep_keys)].copy()

    # Canonicalise target spelling to the report's names, so figures and
    # tables use the same labels as Figure 1.
    canon = {normalise(t): t for t in EXPECTED_TARGETS}
    out["target"] = out["target_raw"].map(lambda t: canon[normalise(t)])
    out = out.drop(columns=["target_raw"])

    # ---- optional: AF3 JSON features -------------------------------------
    if args.af3_dir:
        af3_dir = Path(args.af3_dir)
        print(f"\nparsing AF3 confidence JSONs under {af3_dir} ...")
        af3 = collect_af3_features(af3_dir)
        if not af3.empty:
            before_cols = set(out.columns)
            out = out.merge(af3, on="design", how="left")
            added = sorted(set(out.columns) - before_cols)
            matched = int(out[added[0]].notna().sum()) if added else 0
            print(f"  merged {len(added)} feature(s) onto {matched:,} designs")
        else:
            print("  no AF3 features parsed; continuing with CSV features only")

    # ---- optional: Neff80 per target -------------------------------------
    neff_path = data_dir / args.neff_name
    if neff_path.exists():
        neff = pd.read_csv(neff_path, sep=None, engine="python")
        ncols = {c.lower(): c for c in neff.columns}
        tcol = next((ncols[c] for c in ("target", "name", "receptor")
                     if c in ncols), neff.columns[0])
        vcol = next((ncols[c] for c in ("neff80", "neff", "value")
                     if c in ncols), neff.columns[-1])
        mapping = {normalise(t): v for t, v in zip(neff[tcol], neff[vcol])}
        out["neff80"] = out["target"].map(lambda t: mapping.get(normalise(t)))
        hit = int(out["neff80"].notna().sum())
        print(f"\nmerged Neff80 from {neff_path.name} onto {hit:,} rows")
    else:
        print(f"\nnote: {neff_path.name} not found -- the neff80 feature will be "
              f"absent.  Ask the teammate who ran compute_neff.py for it; it is "
              f"a 13-row file.")

    return out.reset_index(drop=True)


def check_gates(df: pd.DataFrame, strict: bool = True) -> None:
    """Verify the table matches the numbers already published in the report."""
    print("\n" + "=" * 68)
    print("GATES -- the table must agree with the report's Dataset section")
    print("=" * 68)

    results = []

    n = len(df)
    results.append(("designs", n, GATES["n_designs"], n == GATES["n_designs"]))

    n_pos = int(df["label"].sum())
    results.append(("positives", n_pos, GATES["n_positives"],
                    n_pos == GATES["n_positives"]))

    n_t = df["target"].nunique()
    results.append(("targets", n_t, GATES["n_targets"], n_t == GATES["n_targets"]))

    per_t = per_target_scores(df["target"].to_numpy(), df["label"].to_numpy(),
                              df["ipsae_min"].to_numpy(), seed=0)
    mean_ap = float(np.nanmean([v["ap"] for v in per_t.values()]))
    results.append(("mean per-target AP", round(mean_ap, 3),
                    GATES["mean_per_target_ap"],
                    abs(mean_ap - GATES["mean_per_target_ap"]) < 0.01))

    pooled = average_precision(df["label"].to_numpy(), df["ipsae_min"].to_numpy(),
                               n_tiebreaks=200,
                               rng=np.random.default_rng(0))
    results.append(("pooled AP", round(pooled, 3), GATES["pooled_ap"],
                    abs(pooled - GATES["pooled_ap"]) < 0.01))

    for name, got, want, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<22} got {got:<10} "
              f"expected {want}")

    failed = [r[0] for r in results if not r[3]]
    if failed:
        print("\n" + "-" * 68)
        print("Gate failure.  Most likely causes, in order:")
        print("  * the row filter differs -- the report goes 3,669 -> 3,532, so")
        print("    137 designs are removed by the pMHC exclusion.  Check which")
        print("    targets were dropped in the log above.")
        print("  * the label column was read wrongly -- check the positive rate")
        print("    is close to 10.7%.")
        print("  * duplicate constructs: 13 targets map to 16 unique sequences,")
        print("    so some designs may appear more than once.")
        print("Ask the teammate who ran score_e2.py how they filtered; do not")
        print("paper over this with --no-gates.")
        print("-" * 68)
        if strict:
            sys.exit(1)
    else:
        print("\nAll gates passed -- this table is comparable to the ablation "
              "results.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../data",
                    help="directory holding the downloaded deposit")
    ap.add_argument("--csv-name", default="final_dataset.csv")
    ap.add_argument("--neff-name", default="neff80.tsv")
    ap.add_argument("--af3-dir", default=None,
                    help="optional directory of extracted AF3 outputs")
    ap.add_argument("--column-map", default=None,
                    help="optional JSON file overriding column auto-detection")
    ap.add_argument("--out", default="../data/features.csv")
    ap.add_argument("--inspect", action="store_true",
                    help="print the deposit's schema and exit")
    ap.add_argument("--no-gates", action="store_true",
                    help="report gate failures but continue anyway")
    args = ap.parse_args()

    if args.inspect:
        inspect(Path(args.data_dir) / args.csv_name)
        return

    df = build(args)
    check_gates(df, strict=not args.no_gates)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(df):,} rows x {len(df.columns)} columns)")

    # Provenance sidecar, so the report's Methods section can cite exact counts.
    side = out_path.with_suffix(".provenance.json")
    summary = {
        "n_designs": int(len(df)),
        "n_positives": int(df["label"].sum()),
        "positive_rate": round(float(df["label"].mean()), 4),
        "n_targets": int(df["target"].nunique()),
        "targets": sorted(df["target"].unique().tolist()),
        "features": [c for c in df.columns
                     if c not in ("target", "design", "label")],
        "per_target_counts": (df.groupby("target")["label"]
                              .agg(["size", "sum"]).to_dict("index")),
    }
    side.write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    print(f"wrote {side}")

    print("\nfeature columns available to the model:")
    for c in summary["features"]:
        n_missing = int(df[c].isna().sum())
        flag = f"  ({n_missing:,} missing)" if n_missing else ""
        print(f"  - {c}{flag}")


if __name__ == "__main__":
    main()
