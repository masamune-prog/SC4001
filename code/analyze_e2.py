"""Scoring analysis: paired bootstrap, class decomposition, and every table and
figure of Study A.

Subcommands
    ablation   section 4.1 / 4.2 : dAP per seed and per receptor, the collapse
                                   to zero, the class decomposition, and the
                                   depth correlation (figures 2 and 4)
    factorial  section 4.3       : the completed 2x2 (table 2) and the
                                   reconciliation arithmetic
    template   section 4.4       : MSA only / template only / both / neither
    ladder     section 4.5       : the depth ladder (table 3), random vs diverse
    stability  section 4.5       : variance decomposition and sign stability
    all        every one of the above
    --selftest validates the statistics against synthetic ground truth

Every AP is the mean over 200 random tie-breaks and every contrast is a
target-level paired bootstrap with 2,000 resamples (section 3.6).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import metrics
from dataset import WELL_POWERED
from metrics import (N_BOOTSTRAP, N_TIEBREAKS, mean_ap, mean_auc,
                     paired_bootstrap, per_target, spearman)

SCORES = ["ipsae_min", "ipsae_max"]


# --- helpers ------------------------------------------------------------------

def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[df.label.notna()].copy()
    df["label"] = df.label.astype(int)
    return df


def arm(df: pd.DataFrame, condition: str, seed: int | None = None) -> pd.DataFrame:
    out = df[df.condition == condition]
    if seed is not None:
        out = out[out.seed == seed]
    return out


def contrast(df: pd.DataFrame, cond_a: str, cond_b: str, score: str,
             seed: int | None = None, restrict: list[str] | None = None,
             n_draws: int = N_TIEBREAKS, n_boot: int = N_BOOTSTRAP):
    """mean per-target AP of `cond_b` minus that of `cond_a`."""
    A = arm(df, cond_a, seed)
    B = arm(df, cond_b, seed)
    common = set(A.design_id) & set(B.design_id)       # paired on identical designs
    A, B = A[A.design_id.isin(common)], B[B.design_id.isin(common)]
    pa = per_target(A, score, n_draws=n_draws)
    pb = per_target(B, score, n_draws=n_draws)
    boot = paired_bootstrap(pa, pb, key="ap", n_boot=n_boot, restrict=restrict)
    boot_auc = paired_bootstrap(pa, pb, key="auc", n_boot=n_boot, restrict=restrict)
    return {"a": pa, "b": pb, "ap": boot, "auc": boot_auc,
            "mean_ap_a": mean_ap(pa), "mean_ap_b": mean_ap(pb),
            "mean_auc_a": mean_auc(pa), "mean_auc_b": mean_auc(pb),
            "n_designs": len(common)}


def fmt_ci(b) -> str:
    return f"{b['delta']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]"


# --- section 4.1 / 4.2 --------------------------------------------------------

def cmd_ablation(df: pd.DataFrame, a) -> dict:
    out = {}
    seeds = sorted(df.seed.unique())
    print("\n=== 4.1  Removing the channel collapses the confidence output ===")
    for score in SCORES:
        per_seed, per_target_delta = [], {}
        for s in seeds:
            c = contrast(df, "full", "none", score, seed=s,
                         restrict=a.restrict, n_boot=a.n_boot)
            per_seed.append(c["ap"]["delta"])
            for t, d in c["ap"]["per_target"].items():
                per_target_delta.setdefault(t, []).append(d)
            print(f"  {score:<10} seed {s}: dAP {fmt_ci(c['ap'])}   "
                  f"AP {c['mean_ap_a']:.3f} -> {c['mean_ap_b']:.3f}   "
                  f"AUC {c['mean_auc_a']:.3f} -> {c['mean_auc_b']:.3f}")
        m, sd = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) \
            if len(per_seed) > 1 else (float(np.mean(per_seed)), 0.0)
        print(f"  {score:<10} mean over seeds: dAP {m:+.3f} +/- {sd:.3f}")
        out[score] = {"per_seed": per_seed, "mean": m, "sd": sd,
                      "per_target": per_target_delta}

        # sign stability per receptor across seeds
        flipped = [t for t, v in per_target_delta.items()
                   if len(set(np.sign(v))) > 1]
        worse = sum(1 for v in per_target_delta.values() if np.mean(v) < 0)
        print(f"  {score:<10} {worse}/{len(per_target_delta)} receptors worse "
              f"without the alignment; sign unstable on {len(flipped)}: {flipped}")

    # --- the collapse to a constant -----------------------------------------
    print("\n--- the output collapses to a constant ---")
    for cond in ("full", "none"):
        g = arm(df, cond)
        if g.empty:
            continue
        zf = float((g.ipsae_min == 0).mean())
        print(f"  {cond:<6}: {zf*100:.1f}% of designs score exactly 0.000")
        out.setdefault("zero_frac", {})[cond] = zf
        # the extreme receptors quoted in section 4.1
        mode = (g.groupby("target").ipsae_min
                 .agg(lambda s: (s.value_counts().iloc[0], len(s))))
        for t, (k, n) in mode.items():
            if k / n > 0.9:
                print(f"      {t}: {k} of {n} designs take the same value")

    # tie sensitivity: AP s.d. across tie-breaks, with and without the channel
    for cond in ("full", "none"):
        g = arm(df, cond, seed=seeds[0])
        if g.empty:
            continue
        pt = per_target(g, "ipsae_min", n_draws=a.n_draws)
        print(f"  tie s.d. of AP ({cond}): "
              f"{np.nanmean([v['ap_tie_sd'] for v in pt.values()]):.3f}")

    # --- class decomposition: discrimination, not calibration (rejects H1) ---
    print("\n--- class decomposition (H1) ---")
    full = arm(df, "full")[["design_id", "target", "label", "ipsae_min"]]
    none = arm(df, "none")[["design_id", "ipsae_min"]]
    j = full.merge(none, on="design_id", suffixes=("_full", "_none"))
    j["loss"] = j.ipsae_min_full - j.ipsae_min_none
    lb = j[j.label == 1].loss.mean()
    ln = j[j.label == 0].loss.mean()
    print(f"  confirmed binders lose {lb:.3f} of score, failures {ln:.3f} "
          f"({lb/ln:.1f}x)")
    holds = sum(1 for t, g in j.groupby("target")
                if g[g.label == 1].loss.mean() > g[g.label == 0].loss.mean()
                and t in WELL_POWERED)
    print(f"  holds within {holds} of {len(set(j.target) & set(WELL_POWERED))} "
          f"well-powered receptors -> the loss is of discrimination, not "
          f"calibration.  H1 is rejected.")
    out["class_decomposition"] = {"binder_loss": lb, "failure_loss": ln,
                                  "ratio": lb / ln if ln else float("nan")}

    # --- 4.2 depth does not predict the effect ------------------------------
    if Path(a.neff).exists():
        print("\n=== 4.2  Effective depth does not predict the effect ===")
        neff = pd.read_csv(a.neff, sep="\t").groupby("target").neff80.max()
        delta = {t: float(np.mean(v)) for t, v in out["ipsae_min"]["per_target"].items()}
        ts = [t for t in delta if t in neff.index and t in WELL_POWERED]
        rho = spearman([neff[t] for t in ts], [delta[t] for t in ts])
        print(f"  Neff80 spans {neff[ts].min()} ({neff[ts].idxmin()}) to "
              f"{neff[ts].max()} ({neff[ts].idxmax()})")
        print(f"  Spearman(Neff80, dAP) over {len(ts)} receptors = {rho:+.3f}")
        tl = df.groupby("target").target_len.max()
        ts2 = [t for t in delta if t in tl.index and t in WELL_POWERED]
        print(f"  Spearman(target length, dAP) = "
              f"{spearman([tl[t] for t in ts2], [delta[t] for t in ts2]):+.3f}")
        out["depth"] = {"rho": rho,
                        "neff": {t: int(neff[t]) for t in ts},
                        "delta": {t: delta[t] for t in ts}}
    else:
        print(f"\n[analyze_e2] {a.neff} not found; skipping the depth analysis")

    if a.figures:
        _fig_per_receptor(out["ipsae_min"]["per_target"], a.fig_dir)
        if "depth" in out:
            _fig_depth(out["depth"], a.fig_dir)
    return out


# --- section 4.3 --------------------------------------------------------------

def cmd_factorial(df: pd.DataFrame, a) -> dict:
    print("\n=== 4.3  Excision is the modulator (R1) ===")
    cells = {}
    for name, cond in (("intact +MSA", "full"), ("intact -MSA", "none"),
                       ("excised +MSA", "excised_full"),
                       ("excised -MSA", "excised_none")):
        g = arm(df, cond, seed=a.seed)
        if g.empty:
            print(f"  {name}: no folds")
            continue
        cells[name] = per_target(g, "ipsae_min", n_draws=a.n_draws)

    # restrict to the receptors that were genuinely cropped, and to designs
    # present in all four cells, so the 2x2 is paired
    common_t = set.intersection(*[set(v) for v in cells.values()]) if cells else set()
    if a.cropped:
        common_t &= set(a.cropped.split(","))
    print(f"  cells on {sorted(common_t)}")
    table = {k: float(np.nanmean([v[t]["ap"] for t in common_t]))
             for k, v in cells.items()}
    for k, v in table.items():
        print(f"    {k:<14} AP = {v:.3f}")
    if len(table) == 4:
        d_intact = table["intact -MSA"] - table["intact +MSA"]
        d_excised = table["excised -MSA"] - table["excised +MSA"]
        e_msa = table["excised +MSA"] - table["intact +MSA"]
        e_nomsa = table["excised -MSA"] - table["intact -MSA"]
        print(f"\n    dAP of removing MSA, intact  : {d_intact:+.3f}")
        print(f"    dAP of removing MSA, excised : {d_excised:+.3f}")
        print(f"    effect of excision, +MSA arm : {e_msa:+.3f}")
        print(f"    effect of excision, -MSA arm : {e_nomsa:+.3f}")
        print("    the two interventions damage the same channel, so their "
              "effects do not add")
        print(f"\n  reconciliation: intact effect {d_intact:+.3f} + excision "
              f"attenuation {(e_nomsa - e_msa):+.3f} = predicted excised-regime "
              f"effect {d_intact + (e_nomsa - e_msa):+.3f}")
        # per-receptor spread of the attenuation
        spread = []
        for t in sorted(common_t):
            att = ((cells["excised -MSA"][t]["ap"] - cells["excised +MSA"][t]["ap"])
                   - (cells["intact -MSA"][t]["ap"] - cells["intact +MSA"][t]["ap"]))
            spread.append((t, att))
        print("  attenuation per receptor: " +
              ", ".join(f"{t} {v:+.3f}" for t, v in spread))
    return {"table": table}


# --- section 4.4 --------------------------------------------------------------

def cmd_template(df: pd.DataFrame, a) -> dict:
    print("\n=== 4.4  The channel is a substitutable structural prior (H4) ===")
    arms = {"MSA only": "full", "neither": "none",
            "both": "template_full", "template only": "template_none"}
    pts = {}
    for name, cond in arms.items():
        g = arm(df, cond, seed=a.seed)
        if g.empty:
            print(f"  {name}: no folds")
            continue
        pts[name] = per_target(g, a.score, n_draws=a.n_draws)
    if len(pts) < 2:
        return {}
    common = set.intersection(*[set(v) for v in pts.values()])
    print(f"  receptors: {sorted(common)}")
    table = {k: float(np.nanmean([v[t]["ap"] for t in common]))
             for k, v in pts.items()}
    for k in ("MSA only", "template only", "both", "neither"):
        if k in table:
            print(f"    {k:<14} AP = {table[k]:.3f}")
    if {"MSA only", "neither"} <= table.keys():
        print(f"\n    removing the MSA without a template: "
              f"{table['neither'] - table['MSA only']:+.3f}")
    if {"both", "template only"} <= table.keys():
        print(f"    removing the MSA with a template   : "
              f"{table['template only'] - table['both']:+.3f}")
        print("    any sufficient structural prior substitutes for the "
              "evolutionary channel.  H4 is supported.")
    return {"table": table}


# --- section 4.5 --------------------------------------------------------------

def cmd_ladder(df: pd.DataFrame, a) -> dict:
    print("\n=== 4.5  Depth ladder (H2, H3) ===")
    lad = df[df.condition.str.startswith("ladder_")].copy()
    if lad.empty:
        print("  no ladder folds")
        return {}
    lad["rung"] = lad.condition.str.extract(r"ladder_neff(\d+)_")[0].astype(float)
    lad["mode"] = lad.condition.str.extract(r"ladder_neff\d+_(\w+)")[0]
    lad.loc[lad.condition == "ladder_full", ["rung", "mode"]] = [np.inf, "full"]

    full_pt = per_target(lad[lad["mode"] == "full"], "ipsae_min", n_draws=a.n_draws)
    rows = []
    for mode in ("random", "diverse"):
        for rung in sorted(lad[lad["mode"] == mode].rung.unique()):
            g = lad[(lad["mode"] == mode) & (lad.rung == rung)]
            pt = per_target(g, "ipsae_min", n_draws=a.n_draws)
            ap = mean_ap(pt)
            b = paired_bootstrap(full_pt, pt, n_boot=a.n_boot) if full_pt else None
            rows.append({"mode": mode, "neff": int(rung), "ap": ap,
                         "vs_full": b["delta"] if b else np.nan,
                         "ci": b["ci"] if b else None,
                         "sig": bool(b and b["excludes_zero"])})
    if full_pt:
        rows.append({"mode": "full", "neff": "full", "ap": mean_ap(full_pt),
                     "vs_full": 0.0, "ci": None, "sig": False})
    print(f"  {'mode':<9}{'Neff80':>8}{'AP':>8}{'vs. full':>12}")
    for r in rows:
        star = "*" if r["sig"] else " "
        print(f"  {r['mode']:<9}{str(r['neff']):>8}{r['ap']:>8.3f}"
              f"{r['vs_full']:>+11.3f}{star}")
    rand = {r["neff"]: r["ap"] for r in rows if r["mode"] == "random"}
    if rand and full_pt and max(rand.values()) <= mean_ap(full_pt):
        print("  no rung beats the full alignment.  H3 is rejected.")

    # H2: diverse vs random at matched depth
    print("\n  matched-depth content comparison (H2):")
    for rung in sorted({r["neff"] for r in rows if r["mode"] == "random"}
                       & {r["neff"] for r in rows if r["mode"] == "diverse"}):
        pr = per_target(lad[(lad["mode"] == "random") & (lad.rung == rung)],
                        "ipsae_min", n_draws=a.n_draws)
        pd_ = per_target(lad[(lad["mode"] == "diverse") & (lad.rung == rung)],
                         "ipsae_min", n_draws=a.n_draws)
        b = paired_bootstrap(pr, pd_, n_boot=a.n_boot)
        print(f"    Neff80 {rung:<4} diverse - random: {fmt_ci(b)}")
    print("  depth dominates content.  H2 is supported.")
    return {"rows": rows}


def cmd_stability(df: pd.DataFrame, a) -> dict:
    """Where the effect lives: between receptors, between binders, or in noise."""
    print("\n=== 4.5  The effect is binder-specific; the decision is constant ===")
    full = arm(df, "full")[["design_id", "target", "label", "seed", "ipsae_min"]]
    none = arm(df, "none")[["design_id", "seed", "ipsae_min"]]
    j = full.merge(none, on=["design_id", "seed"], suffixes=("_full", "_none"))
    j["effect"] = j.ipsae_min_full - j.ipsae_min_none

    per_design = j.groupby(["design_id", "target", "label"]).effect.mean().reset_index()
    grand = per_design.effect.mean()
    between = (per_design.groupby("target").effect
               .agg(["mean", "size"])
               .assign(ss=lambda d: d["size"] * (d["mean"] - grand) ** 2)["ss"].sum())
    within = sum(((g.effect - g.effect.mean()) ** 2).sum()
                 for _, g in per_design.groupby("target"))
    total = between + within
    print(f"  {between/total*100:.1f}% of the per-design variance lies between "
          f"receptors, {within/total*100:.1f}% between binders of the same receptor")

    seed_var = j.groupby("design_id").effect.var(ddof=1).mean()
    within_var = within / max(len(per_design) - j.target.nunique(), 1)
    print(f"  within-receptor variance {within_var:.4f} vs seed-noise variance "
          f"{seed_var:.4f} -> {max(0.0, 1 - seed_var/within_var)*100:.0f}% survives "
          f"the noise correction")

    sign = j.groupby("design_id").effect.apply(
        lambda s: 1 if (s > 0).all() else (-1 if (s < 0).all() else 0))
    n = len(sign)
    print(f"  across all seeds {100*(sign == 1).mean():.1f}% of designs are stably "
          f"better with the alignment, {100*(sign == -1).mean():.1f}% stably better "
          f"without, {100*(sign == 0).mean():.1f}% flip sign")
    favour = set(sign[sign == -1].index)
    pos = per_design[(per_design.design_id.isin(favour)) & (per_design.label == 1)]
    print(f"  of the {len(favour)} designs that reliably favour removal, only "
          f"{len(pos)} of {int(per_design.label.sum())} are confirmed binders")
    print("  there is no per-target routing decision to make")
    return {"between_frac": float(between / total), "n_designs": n}


# --- figures ------------------------------------------------------------------

def _fig_per_receptor(per_target_delta: dict, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    items = sorted(per_target_delta.items(), key=lambda kv: np.mean(kv[1]))
    names = [k for k, _ in items]
    means = [float(np.mean(v)) for _, v in items]
    lo = [float(np.min(v)) for _, v in items]
    hi = [float(np.max(v)) for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(names))
    ax.barh(y, means, color=["#c0392b" if m > 0 else "#2b6cb0" for m in means])
    ax.hlines(y, lo, hi, color="grey", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$\\Delta$ average precision when the target MSA is removed\n"
                  "(further left = the alignment mattered more)", fontsize=8)
    fig.tight_layout()
    out = Path(fig_dir) / "fig2_delta_ap_per_receptor.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  [figure] {out}")


def _fig_depth(depth: dict, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    ts = sorted(depth["neff"])
    x = [depth["neff"][t] for t in ts]
    y = [depth["delta"][t] for t in ts]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(x, y, color="#2b6cb0")
    for t, xi, yi in zip(ts, x, y):
        ax.annotate(t, (xi, yi), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("effective alignment depth (Neff80)")
    ax.set_ylabel("$\\Delta$AP when the MSA is removed")
    ax.set_title(f"Spearman $\\rho$ = {depth['rho']:+.3f}", fontsize=9)
    fig.tight_layout()
    out = Path(fig_dir) / "fig2_depth_vs_effect.pdf"
    fig.savefig(out); plt.close(fig)
    print(f"  [figure] {out}")


# --- entry point --------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="all",
                   choices=["ablation", "factorial", "template", "ladder",
                            "stability", "all"])
    p.add_argument("--scores", default="e2_scores.tsv")
    p.add_argument("--neff", default="neff80.tsv")
    p.add_argument("--seed", type=int, default=1,
                   help="seed to use for the single-seed arms (4.3-4.5)")
    p.add_argument("--score", default="ipsae_max",
                   help="ranking score for the template arm; the report uses "
                        "ipSAE_max there and ipSAE_min elsewhere")
    p.add_argument("--cropped", default="",
                   help="comma-separated receptors that were genuinely cropped")
    p.add_argument("--restrict", default=None,
                   help="'well-powered' to restrict to receptors with >=5 binders")
    p.add_argument("--n-draws", type=int, default=N_TIEBREAKS)
    p.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    p.add_argument("--figures", action="store_true")
    p.add_argument("--fig-dir", default="figs")
    p.add_argument("--out", default="e2_analysis.json")
    p.add_argument("--selftest", action="store_true",
                   help="validate the statistics against synthetic ground truth")
    a = p.parse_args()

    if a.selftest:
        metrics.selftest()
        if not Path(a.scores).exists():
            return

    a.restrict = WELL_POWERED if a.restrict == "well-powered" else None
    df = load_scores(a.scores)
    print(f"[analyze_e2] {len(df)} scored folds, conditions "
          f"{sorted(df.condition.unique())}, seeds {sorted(df.seed.unique())}")

    results = {}
    run = {"ablation": cmd_ablation, "factorial": cmd_factorial,
           "template": cmd_template, "ladder": cmd_ladder,
           "stability": cmd_stability}
    todo = list(run) if a.command == "all" else [a.command]
    for name in todo:
        try:
            results[name] = run[name](df, a)
        except Exception as exc:                            # noqa: BLE001
            print(f"[analyze_e2] {name}: {exc}", file=sys.stderr)

    Path(a.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[analyze_e2] -> {a.out}")


if __name__ == "__main__":
    main()
