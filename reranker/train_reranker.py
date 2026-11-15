"""
Train the MLP re-ranker and produce out-of-fold predictions.

The model sits on top of a frozen AlphaFold 3: it never touches AF3's weights,
it only re-orders designs using AF3's own confidence outputs plus a few
properties of the inputs.  The question it answers is narrow -- is there signal
in the AF3 output vector that a single scalar (ipSAE) throws away?

Design decisions, and why
-------------------------
* CPU only.  3,532 rows x ~12 features is a few thousand parameters.  A GPU
  would be slower than the CPU here because the transfer costs more than the
  arithmetic.  Do not install ROCm for this.

* Leave-one-target-out (LOTO).  Designs for the same receptor are not
  independent, so a random split would leak target identity and flatter the
  model.  Thirteen folds, each holding out one whole receptor.

* Within-target feature standardisation.  AP is computed inside a target, so
  only the *ordering* within a target matters.  Raw ipSAE values carry a large
  per-target offset (easy targets score high across the board); left in, the
  model spends its capacity learning those offsets, which cannot transfer to a
  held-out receptor.  Standardising each feature within its own target removes
  the offset and leaves the within-target ranking signal.
  This uses the held-out target's own designs to compute its mean and s.d.  That
  is legitimate -- in a real campaign the whole candidate set for a target is in
  hand before anything is synthesised -- but it is transductive, and the report
  says so explicitly rather than hiding it.

* Two losses.  BCE treats the task as classification; the pairwise logistic
  loss (RankNet-style, pairs drawn within a target) optimises the ordering that
  AP actually measures.  Both are implemented so the report can state which was
  chosen and what the other did.

* Several model seeds per fold, averaged.  A small MLP on 3k rows is seed-
  sensitive; one seed would report noise.

Usage
-----
    python train_reranker.py --features ../data/features.csv \
        --feature-set af3 --loss pairwise --out ../results/oof_af3.csv

    python train_reranker.py --features ../data/features.csv \
        --feature-set af3+input --loss pairwise --out ../results/oof_full.csv

Writes one row per design with the out-of-fold prediction, ready for
eval_reranker.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from metrics import average_precision

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

# --------------------------------------------------------------------------
# Feature sets -- these define the rows of Table 5 in the report
# --------------------------------------------------------------------------

AF3_FEATURES = [
    "ipsae_min", "ipsae_max", "iptm", "ptm", "plddt", "pae_interaction",
    # present only when extract_features.py was given --af3-dir
    "af3_iptm", "af3_ptm", "af3_ranking_score", "af3_fraction_disordered",
    "plddt_mean", "plddt_min", "plddt_frac_above_70",
    "pae_interface_mean", "pae_interface_min", "pae_interface_frac_below_10",
    "n_interface_pairs", "ipsae_min_recomputed", "ipsae_max_recomputed",
    # present when the table came from build_features_from_deposit.py
    "ipsae_avg", "iptm_avg", "iptm_std", "lis",
    "pdockq_min", "pdockq_max", "pdockq2_min", "pdockq2_max",
    "min_pae_contact", "max_pae_contact", "n_contact_res",
    "ipsae_d0chn", "ipsae_n_in_calc",
    # geometry and energetics of AF3's PREDICTED complex
    "af3_delta_sasa", "af3_nonpolar_count", "af3_percent_nonpolar",
    "af3_hbonds", "af3_n_interface_res", "af3_interface_dG",
    "af3_interface_sc", "af3_interface_packstat", "af3_interface_dsasa",
    "af3_dG_sasa_ratio", "af3_unsat_hbonds", "af3_sap_delta",
    "af3_binder_score", "rmsd_input_af3", "rmsd_chA_input_af3",
    # present when the table came from recompute_ipsae_from_folds.py
    "contact_prob_max", "contact_prob_sum", "plddt_binder_mean",
    "plddt_target_mean", "chain_pair_iptm_ab", "chain_pair_pae_min_ab",
    "af3_has_clash",
]

INPUT_FEATURES = [
    "binder_len", "target_len", "neff80",
    # design-side descriptors, computed before any structure prediction
    "input_interface_dG", "input_interface_sc", "input_interface_packstat",
    "input_interface_dsasa", "input_dG_sasa_ratio", "input_unsat_hbonds",
    "input_binder_score", "input_surface_hydrophobicity", "input_sap_delta",
    "input_delta_sasa", "input_hbonds", "input_percent_nonpolar",
    "input_n_interface_res", "camsol_score",
]

# Features describing the no-MSA condition.  Only available once the ablation
# scores are exported; this is the variant that ties Experiment 4 to 1-3.
NOMSA_FEATURES = ["ipsae_min_nomsa", "ipsae_max_nomsa",
                  "delta_ipsae_min", "delta_ipsae_max"]

# The five confidence scores a practitioner would reach for first.  Kept as a
# named set so the report's feature-count comparison -- widening the input from
# five features to the full AF3 block makes the model worse, not better -- is
# reproducible from a single flag rather than from an edit to the list above.
AF3_CORE_FEATURES = [
    "ipsae_min", "ipsae_max", "iptm", "ptm", "pae_interaction",
]

FEATURE_SETS = {
    "af3core": AF3_CORE_FEATURES,
    "af3": AF3_FEATURES,
    "af3+input": AF3_FEATURES + INPUT_FEATURES,
    "af3+input+nomsa": AF3_FEATURES + INPUT_FEATURES + NOMSA_FEATURES,
}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class MLPReranker(nn.Module):
    """A deliberately small MLP.

    With 391 positives spread over 13 targets, anything larger memorises.  Two
    hidden layers at 32 and 16 units is roughly 1-2k parameters depending on
    the feature set -- small enough that LOTO is meaningful.
    """

    def __init__(self, d_in: int, hidden=(32, 16), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def bce_loss(scores, labels, pos_weight):
    """Binary cross-entropy with the positive class up-weighted.

    The benchmark is 10.7% positive, so without a weight the model can reach a
    low loss by predicting 'no binder' everywhere.
    """
    return nn.functional.binary_cross_entropy_with_logits(
        scores, labels, pos_weight=pos_weight
    )


def pairwise_loss(scores, labels, groups, rng, max_pairs: int = 4096):
    """RankNet-style logistic loss over (positive, negative) pairs.

    Pairs are drawn only *within* a target, because AP is computed within a
    target -- comparing a design for EGFR against one for Mdm2 is not a
    comparison the metric ever makes.
    """
    losses = []
    for g in torch.unique(groups):
        mask = groups == g
        s = scores[mask]
        y = labels[mask]
        pos = torch.nonzero(y > 0.5, as_tuple=True)[0]
        neg = torch.nonzero(y < 0.5, as_tuple=True)[0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        n_pairs = min(max_pairs, pos.numel() * neg.numel())
        pi = pos[torch.from_numpy(rng.integers(0, pos.numel(), n_pairs))]
        ni = neg[torch.from_numpy(rng.integers(0, neg.numel(), n_pairs))]
        # softplus(-(s_pos - s_neg)) -> 0 when the positive outranks the negative
        losses.append(nn.functional.softplus(-(s[pi] - s[ni])).mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def standardise_within_target(df: pd.DataFrame, features: list[str]
                              ) -> np.ndarray:
    """Z-score each feature inside each target; see the module docstring."""
    out = np.zeros((len(df), len(features)), dtype=np.float32)
    for j, feat in enumerate(features):
        col = df[feat].astype(float)
        grouped = col.groupby(df["target"])
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        z = (col - mean) / std
        # A feature constant within a target carries no within-target ranking
        # information; zero is the correct standardised value for it.
        out[:, j] = z.fillna(0.0).to_numpy(dtype=np.float32)
    return out


def standardise_pooled(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Z-score each feature over the whole table, ignoring target identity.

    The ablation control for ``standardise_within_target``.  Pooling leaves the
    per-target score offsets in the data, so a model can lower its training
    loss by learning which receptor a design belongs to -- signal that is
    useless at test time, because the held-out receptor is by construction one
    it has never seen.
    """
    out = np.zeros((len(df), len(features)), dtype=np.float32)
    for j, feat in enumerate(features):
        col = df[feat].astype(float)
        std = col.std()
        z = (col - col.mean()) / (std if std and np.isfinite(std) else np.nan)
        out[:, j] = z.fillna(0.0).to_numpy(dtype=np.float32)
    return out


STANDARDISERS = {
    "within": standardise_within_target,
    "pooled": standardise_pooled,
}


def resolve_features(df: pd.DataFrame, requested: list[str]) -> list[str]:
    """Keep requested features that exist and are usable."""
    keep = []
    for f in requested:
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        if col.notna().sum() < 0.5 * len(df):
            print(f"  skipping {f!r}: more than half the values are missing")
            continue
        if col.nunique(dropna=True) < 2:
            print(f"  skipping {f!r}: constant")
            continue
        keep.append(f)
    return keep


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_one(
    x_tr, y_tr, g_tr, x_va, *, loss_name, epochs, lr, weight_decay,
    hidden, dropout, seed,
):
    """Fit one model and return validation scores."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = MLPReranker(x_tr.shape[1], hidden=hidden, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xt = torch.from_numpy(x_tr)
    yt = torch.from_numpy(y_tr.astype(np.float32))
    gt = torch.from_numpy(g_tr.astype(np.int64))
    n_pos = float(y_tr.sum())
    n_neg = float(len(y_tr) - n_pos)
    pos_weight = torch.tensor(max(n_neg / max(n_pos, 1.0), 1.0))

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        scores = model(xt)
        if loss_name == "bce":
            loss = bce_loss(scores, yt, pos_weight)
        else:
            loss = pairwise_loss(scores, yt, gt, rng)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_va)).numpy()


def run_loto(df, features, args) -> pd.DataFrame:
    """Leave-one-target-out: every design gets a prediction from a model that
    never saw its receptor."""
    x_all = STANDARDISERS[args.standardise](df, features)
    y_all = df["label"].to_numpy()
    targets = df["target"].to_numpy()
    uniq = sorted(set(targets.tolist()))
    group_idx = {t: i for i, t in enumerate(uniq)}
    g_all = np.array([group_idx[t] for t in targets])

    preds = np.full(len(df), np.nan, dtype=np.float64)
    t0 = time.time()

    for fold, held in enumerate(uniq, 1):
        te = targets == held
        tr = ~te
        if y_all[tr].sum() == 0:
            print(f"  fold {fold:>2}/{len(uniq)}  {held:<16} skipped "
                  f"(no positives in training set)")
            continue

        # Average over model seeds: a small MLP on this much data is seed
        # sensitive, and a single seed would report noise as a result.
        fold_preds = []
        for seed in range(args.model_seeds):
            fold_preds.append(train_one(
                x_all[tr], y_all[tr], g_all[tr], x_all[te],
                loss_name=args.loss, epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay,
                hidden=tuple(args.hidden), dropout=args.dropout,
                seed=args.seed * 1000 + seed,
            ))
        preds[te] = np.mean(fold_preds, axis=0)

        ap = average_precision(y_all[te], preds[te],
                               rng=np.random.default_rng(0))
        base = average_precision(y_all[te], df.loc[te, "ipsae_min"].to_numpy(),
                                 rng=np.random.default_rng(0))
        print(f"  fold {fold:>2}/{len(uniq)}  {held:<16} "
              f"n={int(te.sum()):>4}  pos={int(y_all[te].sum()):>3}  "
              f"AP {ap:.3f}  (ipSAE {base:.3f}, delta {ap - base:+.3f})")

    print(f"  {len(uniq)} folds in {time.time() - t0:.1f}s on CPU")

    out = df[["target", "design", "label", "ipsae_min"]].copy()
    if "ipsae_max" in df.columns:
        out["ipsae_max"] = df["ipsae_max"]
    out["pred"] = preds
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="../data/features.csv")
    ap.add_argument("--feature-set", default="af3+input",
                    choices=sorted(FEATURE_SETS))
    ap.add_argument("--loss", default="pairwise", choices=["pairwise", "bce"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, nargs="*", default=[32, 16],
                    help="hidden layer widths; pass with no values "
                         "(--hidden) for a linear model, i.e. no hidden layer")
    ap.add_argument("--standardise", default="within",
                    choices=sorted(STANDARDISERS),
                    help="'within' z-scores each feature inside each target "
                         "(the reported configuration); 'pooled' z-scores "
                         "over the whole table and is the ablation control")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--model-seeds", type=int, default=5,
                    help="models averaged per fold")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="../results/oof_predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    print(f"loaded {len(df):,} designs over {df['target'].nunique()} targets, "
          f"{int(df['label'].sum())} positives")

    print(f"\nresolving feature set {args.feature_set!r}:")
    features = resolve_features(df, FEATURE_SETS[args.feature_set])
    if not features:
        raise SystemExit("no usable features -- check features.csv")
    print(f"  using {len(features)}: {features}")

    # Rows with a missing value in a used feature are filled with the target
    # median, so a single absent column does not discard a whole design.
    for f in features:
        df[f] = pd.to_numeric(df[f], errors="coerce")
        df[f] = df[f].fillna(df.groupby("target")[f].transform("median"))
        df[f] = df[f].fillna(df[f].median())

    print(f"\nleave-one-target-out, loss={args.loss}, "
          f"hidden={args.hidden}, {args.model_seeds} model seeds per fold")
    out = run_loto(df, features, args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    meta = {
        "feature_set": args.feature_set,
        "features_used": features,
        "n_features": len(features),
        "loss": args.loss,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "model_seeds": args.model_seeds,
        "seed": args.seed,
        "n_designs": int(len(df)),
        "n_targets": int(df["target"].nunique()),
        "evaluation": "leave-one-target-out",
        "standardisation": ("per-feature z-score within target"
                            if args.standardise == "within"
                            else "per-feature z-score pooled across targets"),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {meta_path}  (paste these into the report's Methods)")


if __name__ == "__main__":
    main()
