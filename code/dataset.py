"""Loader for the Overath et al. benchmark deposit (Zenodo 15722219).

The deposit's column names are resolved through an alias table so that a small
change in the CSV header does not break every downstream script.  If a required
field cannot be resolved the error message lists the columns that *are* present.

Retention funnel implemented here (section 3.2 of the report):
    3,676 deposited  ->  3,669 with available scores  ->  3,532 against 13 targets
                                                          after excluding pMHC.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys

import pandas as pd

# --- column resolution -------------------------------------------------------

# The deposit's *actual* column names are listed FIRST in each entry, so the
# exact-match pass resolves them before the loose contains-match fallback ever
# runs.  That ordering is load-bearing -- see the two traps below.
#
# Trap 1: chain roles are inverted relative to this project's own convention.
# In the deposit, binder_chain == "A" and target_chains == ["B"], and msa_info
# reads ["A:no_msa", "B:run_msa"] -- the binder is chain A and the *target* is
# chain B.  So binder_seq is A_seq and target_seq is B_seq.  Confirmed by the
# cardinalities: A_seq has 3,669 distinct values (one per design) while B_seq
# has 19 (one per target construct).  Getting this backwards silently swaps the
# ablated channel and every downstream number is meaningless.
#
# Trap 2: "binder" is the deposit's *label* column (True/False), not a
# sequence.  It must not appear among the binder_seq aliases: the old table
# listed it there, and because "binder" is an exact column name it would have
# won the exact-match pass and loaded the label as the binder sequence.
ALIASES = {
    "design_id": ["binder_id", "design_id", "design", "binder_name",
                  "design_name", "name"],
    "target": ["target_id", "target", "target_name", "receptor"],
    # chain A is the binder, chain B is the target -- see Trap 1 above.
    "binder_seq": ["a_seq", "binder_seq", "binder_sequence"],
    "target_seq": ["b_seq", "target_seq", "target_sequence",
                   "receptor_sequence"],
    # "binder" is the deposit's label column -- see Trap 2 above.
    "label": ["binder", "label", "binds", "binder_label",
              "experimental_outcome", "outcome", "success", "is_binder"],
    # scores carried through from the deposit (used by the E0 validation gate)
    "ipsae_min_deposit": ["af3_ipsae_min", "ipsae_min", "ipsae_af3_min"],
    "ipsae_max_deposit": ["af3_ipsae_max", "ipsae_max", "ipsae_af3_max"],
}

# Targets excluded in section 3.2: short presented peptides make alignment
# depth uninterpretable, and they hold only 3 of the 394 confirmed binders.
PMHC_PATTERNS = [r"pmhc", r"hla", r"mhc"]

# The 13 receptors retained, as named in the report's figures/tables.
RETAINED_TARGETS = [
    "EGFR", "FGFR2", "IL2Ra", "IL7Ra", "IL10Ra", "InsulinR", "LTK",
    "Mdm2", "Pdl1", "SARS_CoV2_RBD", "TrkA", "VirB8", "sntx",
]

# Receptors with >= 5 confirmed binders; per-receptor analyses are restricted
# to these (section 6: LTK has 3, IL10Ra has 2).
WELL_POWERED = [t for t in RETAINED_TARGETS if t not in ("LTK", "IL10Ra")]


def resolve(df: pd.DataFrame, key: str, required: bool = True) -> str | None:
    lowered = {c.lower().strip(): c for c in df.columns}
    for cand in ALIASES[key]:
        if cand in lowered:
            return lowered[cand]
    # Fall back to a loose contains-match -- but say so.  A silent loose match
    # is how a schema change turns into wrong numbers rather than a crash: the
    # generic alias "id" happily matches "binder_id", and would just as happily
    # have matched some other *_id column had the header been ordered
    # differently.  Announce every fallback so it can be checked.
    for cand in ALIASES[key]:
        for low, orig in lowered.items():
            if cand in low:
                print(f"[dataset] WARNING {key!r} resolved loosely: alias "
                      f"{cand!r} matched column {orig!r}. Verify this is the "
                      f"intended column.", file=sys.stderr)
                return orig
    if required:
        raise KeyError(
            f"could not resolve column {key!r}; tried {ALIASES[key]}.\n"
            f"columns present: {list(df.columns)}"
        )
    return None


def _normalise_label(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s > 0).astype(int)
    truthy = {"1", "true", "yes", "y", "binder", "bind", "binds", "success",
              "positive", "hit"}
    return s.astype(str).str.strip().str.lower().isin(truthy).astype(int)


def target_uid(sequence: str) -> str:
    """Stable id for a *unique target sequence*.

    13 targets map to 16 sequences because some receptors were deposited with
    several constructs (section 3.2); alignments are built per unique sequence,
    so the sequence hash — not the receptor name — is the alignment key.
    """
    return hashlib.sha1(sequence.strip().upper().encode()).hexdigest()[:10]


def load_dataset(path: str, keep_pmhc: bool = False,
                 require_scores: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = {k: resolve(raw, k, required=k in
                       ("design_id", "target", "binder_seq", "target_seq", "label"))
            for k in ALIASES}

    df = pd.DataFrame({
        "design_id": raw[cols["design_id"]].astype(str).str.strip(),
        "target": raw[cols["target"]].astype(str).str.strip(),
        "binder_seq": raw[cols["binder_seq"]].astype(str).str.strip().str.upper(),
        "target_seq": raw[cols["target_seq"]].astype(str).str.strip().str.upper(),
        "label": _normalise_label(raw[cols["label"]]),
    })
    for extra in ("ipsae_min_deposit", "ipsae_max_deposit"):
        if cols.get(extra):
            df[extra] = pd.to_numeric(raw[cols[extra]], errors="coerce")

    n0 = len(df)
    # 3,676 -> 3,669: drop designs with no deposited scores.
    if require_scores and "ipsae_min_deposit" in df:
        df = df[df["ipsae_min_deposit"].notna()]
    n1 = len(df)

    # 3,669 -> 3,532: drop the two pMHC targets.
    if not keep_pmhc:
        pat = "|".join(PMHC_PATTERNS)
        df = df[~df["target"].str.lower().str.contains(pat, regex=True, na=False)]
    n2 = len(df)

    df = df[(df.binder_seq.str.len() > 0) & (df.target_seq.str.len() > 0)]
    df["target_uid"] = df.target_seq.map(target_uid)
    df = df.reset_index(drop=True)

    print(f"[dataset] {n0} deposited -> {n1} with scores -> {n2} after pMHC "
          f"exclusion -> {len(df)} retained across {df.target.nunique()} targets "
          f"({df.target_uid.nunique()} unique target sequences); "
          f"{int(df.label.sum())} confirmed binders "
          f"({df.label.mean():.3f} positive rate)", file=sys.stderr)
    return df


def carry_through_columns(path: str) -> pd.DataFrame:
    """The raw deposit, unmodified — used for the 16 design-side features."""
    return pd.read_csv(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="inspect the benchmark CSV")
    p.add_argument("--dataset", default="final_dataset.csv")
    p.add_argument("--keep-pmhc", action="store_true")
    a = p.parse_args()
    d = load_dataset(a.dataset, keep_pmhc=a.keep_pmhc)
    print(d.groupby("target").agg(n=("label", "size"), binders=("label", "sum"),
                                  seqs=("target_uid", "nunique")))
