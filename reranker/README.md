# Study B — the learned re-ranker

Trains a small MLP to re-rank designed binders on top of a frozen AlphaFold 3's
confidence outputs, and asks whether it beats ranking by ipSAE alone. This is
Study B of the report; Study A (the input-channel ablation) is the code in
[`../code/`](../code/).

**Headline result: it does not beat ipSAE.** That is the finding, not a failure
to reach one — see "What the numbers say" below.

---

## Why this directory is self-contained

It has its own `metrics.py`, which is **not** the same implementation as
[`../code/metrics.py`](../code/metrics.py). The two expose different APIs:

| | `../code/metrics.py` | `./metrics.py` |
|---|---|---|
| per-target scores | `per_target(df, score_col)` | `per_target_scores(targets, y, scores)` |
| tie handling | `ap_tiebreak(y, s, n_draws)` | `ap_tiebreak_sd(y, s, rng)` |
| AP | `average_precision(y, order)` | `average_precision(y, scores, n_tiebreaks)` |

Both compute the same quantities and agree on the baseline; they were written
at different times against different callers. Rather than refactor Study A's
pipeline days before the deadline — which would mean re-validating every number
in Study A — Study B keeps its own copy and imports nothing from `../code/`.

Python resolves `from metrics import ...` against the script's own directory,
so running anything in here picks up this `metrics.py`, not the other one.
**Run these scripts from inside this directory.**

Merging the two is a sensible post-submission cleanup. It is not a bug.

---

## Dependencies

Python 3.10+. CPU only — the model has fewer than 2,000 parameters and a full
13-fold run takes about 80 seconds. A GPU is not merely unnecessary, it is
slower, because moving the data costs more than the arithmetic.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Data

One public file, from the Overath et al. deposit, Zenodo record `15722219`
(CC-BY-4.0). The repo `.gitignore` deliberately keeps it out of version
control; fetch it into a directory outside this one:

```bash
mkdir -p ../data
curl -L -o ../data/final_dataset.csv \
  https://zenodo.org/api/records/15722219/files/final_dataset.csv/content
```

Optionally, for the higher-fidelity feature path, the deposit's raw AF3
confidence files (2.0 GB compressed):

```bash
curl -L -o ../data/AF3_outputs.tar.zstd \
  https://zenodo.org/api/records/15722219/files/AF3_outputs.tar.zstd/content

mkdir -p ../data/af3_json
zstd -dc ../data/AF3_outputs.tar.zstd | \
  tar -xf - -C ../data/af3_json \
      --wildcards --no-wildcards-match-slash '*/*_confidences.json'
```

The feature builder filters to the 13-receptor benchmark: it drops the two pMHC
targets, leaving **3,532 designs, 391 positives, 13 targets**. It asserts those
three counts and refuses to write its output if any is wrong, so a changed or
truncated download fails loudly rather than quietly producing other numbers.

## Commands

```bash
# features: ipSAE recomputed from the deposit's raw PAE matrices (~25 min)
python recompute_ipsae_from_folds.py \
    --af3-dir ../data/af3_json \
    --dataset ../data/final_dataset.csv \
    --out results/features_folds.csv

# train, leave-one-target-out over the 13 receptors (~80 s each)
python train_reranker.py --features results/features_folds.csv \
    --feature-set af3       --out results/oofF_af3.csv
python train_reranker.py --features results/features_folds.csv \
    --feature-set af3+input --out results/oofF_af3_input.csv

# contrast against the ipSAE baseline; writes the report's Table 5
python eval_reranker.py \
    --oof af3=results/oofF_af3.csv \
    --oof af3+input=results/oofF_af3_input.csv \
    --out-dir results --figure

# the design-ablation grid (capacity / features / loss / normalisation)
./run_ablations.sh
```

Defaults are the reported configuration: pairwise logistic loss, hidden layers
32→16, dropout 0.2, 300 epochs, lr 3e-3, weight decay 1e-2, five model seeds
averaged per fold, 2,000 bootstrap resamples. Every run writes a `.meta.json`
beside its output recording the full configuration and the resolved feature
list, so any saved result can be traced back to the command that made it.

## What the numbers say

| Method | Mean per-target AP | Mean AUC | ΔAP vs ipSAE_min |
|---|---|---|---|
| ipSAE_min (AF3, full MSA) | 0.549 | 0.789 | — |
| MLP, AF3 outputs only | 0.481 | 0.774 | −0.067 [−0.144, −0.009] |
| MLP, AF3 + input features | 0.545 | 0.809 | −0.004 [−0.071, +0.074] |

The re-ranker matches ipSAE and does not beat it. It *does* raise mean ROC-AUC
(0.789 → 0.809), which is the informative part: AUC integrates the whole
ranking while AP is dominated by the top, so the model learns to order the bulk
of the non-binders better without promoting any additional true binder into the
region a campaign would actually synthesise.

**Why it cannot do better.** The unit of replication is the receptor, so
leave-one-target-out gives 13 training folds regardless of the 3,532 designs —
the effective sample size for generalisation is 13, not 3,532. The ablation
grid measures this directly: widening the feature set buys variance, not
signal.

## A note on two baselines

Study B's baseline is **0.549**, computed on the deposit's published AF3 folds.
Study A quotes **0.517** for the same quantity, computed on our own folds
(`../all_af3_folds.csv`, `experiment == "e2_out"`, `condition == "full"`, which
reproduces 0.5196; the residual is tie-break randomisation).

Both are correct. They differ because they are different sets of folds, run
against different database vintages. All three rows of the table above share
the one baseline, so the contrasts inside the table are internally consistent;
only the absolute values are not comparable with Study A's tables.
