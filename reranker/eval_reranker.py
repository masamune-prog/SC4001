"""
Evaluate the re-ranker against the ipSAE baseline and emit report-ready output.

Uses exactly the protocol the rest of the report uses, so the numbers sit
beside the ablation results without a caveat:

  * AP computed within each target, then averaged over targets
  * 200 random tie-breaks per AP (the ipSAE baseline is heavily tied)
  * target-level paired bootstrap, 2,000 resamples, for every contrast

Outputs
-------
  results_summary.tsv      per-method AP / AUC and the bootstrap contrast
  per_target.tsv           per-receptor breakdown, for the failure-case text
  table5.tex               the filled Table 5, ready to paste into Draft.tex
  fig5_reranker.pdf        optional per-receptor delta plot (--figure)

Usage
-----
    python eval_reranker.py \
        --oof af3=../results/oof_af3.csv \
        --oof af3+input=../results/oof_full.csv \
        --out-dir ../results --figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import (ap_tiebreak_sd, paired_bootstrap, per_target_scores)

# Labels used in the report's Table 5, keyed by --oof name.
ROW_LABELS = {
    "af3": r"MLP, AF3 outputs only",
    "af3+input": r"MLP, AF3 outputs $+$ input features",
    "af3+input+nomsa": r"MLP, $+$ no-MSA condition features",
}


def evaluate(df: pd.DataFrame, score_col: str, min_positives: int, seed: int):
    return per_target_scores(
        df["target"].to_numpy(),
        df["label"].to_numpy(),
        df[score_col].to_numpy(),
        min_positives=min_positives,
        seed=seed,
    )


def fmt_ci(point: float, lo: float, hi: float) -> str:
    """Signed point estimate with its interval, as LaTeX.

    Each number is wrapped in its own math group so that a leading minus
    renders as a proper minus sign.  Do not try to emit "$-$" inside an
    enclosing $...$ -- that closes math mode and breaks the build.
    """
    return f"${point:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", action="append", required=True, metavar="NAME=PATH",
                    help="out-of-fold predictions, repeatable "
                         "(e.g. af3=../results/oof_af3.csv)")
    ap.add_argument("--baseline-col", default="ipsae_min")
    ap.add_argument("--min-positives", type=int, default=1,
                    help="per-target AP needs positives; the report uses >=5 "
                         "for per-receptor analyses")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="../results")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    runs: dict[str, pd.DataFrame] = {}
    for spec in args.oof:
        if "=" not in spec:
            raise SystemExit(f"--oof expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        runs[name] = pd.read_csv(path)

    first = next(iter(runs.values()))
    base_per_t = evaluate(first, args.baseline_col, args.min_positives, args.seed)
    base_ap = {k: v["ap"] for k, v in base_per_t.items()}
    base_mean = float(np.nanmean(list(base_ap.values())))
    base_auc = float(np.nanmean([v["auc"] for v in base_per_t.values()]))

    # Tie sensitivity of the baseline, quoted as a diagnostic in the report.
    tie_sd = float(np.nanmean([
        ap_tiebreak_sd(first.loc[first["target"] == t, "label"].to_numpy(),
                       first.loc[first["target"] == t, args.baseline_col].to_numpy(),
                       rng=np.random.default_rng(0))
        for t in base_per_t
    ]))

    print("=" * 72)
    print(f"Baseline: {args.baseline_col}")
    print(f"  mean per-target AP   {base_mean:.3f}   "
          f"(report quotes 0.517 for ipSAE_min)")
    print(f"  mean per-target AUC  {base_auc:.3f}")
    print(f"  tie-break s.d.       {tie_sd:.3f}")
    print(f"  targets evaluated    {len(base_ap)}")
    if abs(base_mean - 0.517) > 0.02:
        print("  WARNING: baseline differs from the report's 0.517 by more than")
        print("           0.02 -- re-check the row filter before trusting any")
        print("           of the comparisons below.")
    print("=" * 72)

    rows = []
    per_target_rows = []
    for t, v in base_per_t.items():
        per_target_rows.append({"target": t, "n": v["n"], "n_pos": v["n_pos"],
                                "method": "ipsae_min", "ap": v["ap"],
                                "auc": v["auc"]})

    for name, df in runs.items():
        per_t = evaluate(df, "pred", args.min_positives, args.seed)
        model_ap = {k: v["ap"] for k, v in per_t.items()}
        mean_ap = float(np.nanmean(list(model_ap.values())))
        mean_auc = float(np.nanmean([v["auc"] for v in per_t.values()]))
        point, lo, hi = paired_bootstrap(base_ap, model_ap,
                                         n_resamples=args.bootstrap,
                                         seed=args.seed)
        sig = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"\n{name}")
        print(f"  mean per-target AP   {mean_ap:.3f}")
        print(f"  mean per-target AUC  {mean_auc:.3f}")
        print(f"  delta AP vs baseline {point:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]   CI excludes zero: {sig}")
        n_better = sum(1 for k in model_ap if model_ap[k] > base_ap.get(k, np.nan))
        print(f"  receptors improved   {n_better}/{len(model_ap)}")

        rows.append({"method": name, "mean_ap": mean_ap, "mean_auc": mean_auc,
                     "delta_ap": point, "ci_low": lo, "ci_high": hi,
                     "significant": sig, "n_better": n_better,
                     "n_targets": len(model_ap)})
        for t, v in per_t.items():
            per_target_rows.append({"target": t, "n": v["n"], "n_pos": v["n_pos"],
                                    "method": name, "ap": v["ap"], "auc": v["auc"]})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [{"method": "ipsae_min", "mean_ap": base_mean, "mean_auc": base_auc,
          "delta_ap": 0.0, "ci_low": np.nan, "ci_high": np.nan,
          "significant": "-", "n_better": np.nan, "n_targets": len(base_ap)}]
        + rows)
    summary.to_csv(out_dir / "results_summary.tsv", sep="\t", index=False)

    pt = pd.DataFrame(per_target_rows)
    pt.to_csv(out_dir / "per_target.tsv", sep="\t", index=False)

    # ---- Table 5, filled ------------------------------------------------
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Re-ranking under leave-one-target-out evaluation. "
        r"Average precision is computed within each target and averaged over "
        r"targets; intervals are target-level paired bootstraps with 2,000 "
        r"resamples.}",
        r"  \label{tab:mlp}",
        r"  \small",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Method & Mean per-target AP $\uparrow$ & \dAP{} vs.\ \ipsaemin{} "
        r"[95\% CI]\\",
        r"    \midrule",
        rf"    \ipsaemin{{}} (AF3, full MSA) & {base_mean:.3f} & ---\\",
    ]
    for r in rows:
        label = ROW_LABELS.get(r["method"], r["method"].replace("_", r"\_"))
        lines.append(f"    {label} & {r['mean_ap']:.3f} & "
                     f"{fmt_ci(r['delta_ap'], r['ci_low'], r['ci_high'])}\\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (out_dir / "table5.tex").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {out_dir / 'results_summary.tsv'}")
    print(f"wrote {out_dir / 'per_target.tsv'}")
    print(f"wrote {out_dir / 'table5.tex'}   <- paste over Table 5 in Draft.tex")

    # ---- worst and best receptors, for the failure-case paragraph -------
    if rows:
        best_name = max(rows, key=lambda r: r["delta_ap"])["method"]
        sub = pt[pt["method"] == best_name].set_index("target")
        base_idx = pt[pt["method"] == "ipsae_min"].set_index("target")
        delta = (sub["ap"] - base_idx["ap"]).sort_values()
        print(f"\nper-receptor deltas for {best_name} "
              f"(use these for the failure-case paragraph):")
        for t, d in delta.items():
            print(f"  {t:<16} {d:+.3f}   "
                  f"(n={int(base_idx.loc[t, 'n'])}, "
                  f"pos={int(base_idx.loc[t, 'n_pos'])})")

    # ---- optional figure -------------------------------------------------
    if args.figure and rows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not installed; skipping --figure")
            return

        best_name = max(rows, key=lambda r: r["delta_ap"])["method"]
        sub = pt[pt["method"] == best_name].set_index("target")
        base_idx = pt[pt["method"] == "ipsae_min"].set_index("target")
        delta = (sub["ap"] - base_idx["ap"]).sort_values()

        fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(delta) + 1.6))
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in delta.values]
        ax.barh(range(len(delta)), delta.values, color=colors)
        ax.set_yticks(range(len(delta)))
        ax.set_yticklabels(delta.index)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel(r"$\Delta$ average precision, MLP re-ranker $-$ ipSAE$_{\min}$"
                      "\n(right = the re-ranker helped)")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = out_dir / "fig5_reranker.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
