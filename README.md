# Does AlphaFold 3 Need Evolution to Recognise a Binder?

SC4001 project (NTU, Y4S1). AlphaFold 3 is held frozen throughout; the question
is how much of its ability to separate real binders from decoys survives when
the evolutionary signal available to it is removed or degraded.

Every fold is run under matched conditions and scored with ipSAE, so a pair of
conditions differs only in what the model was allowed to see:

| Condition family | What changes |
| --- | --- |
| `full` / `none` | Target MSA present vs. stripped |
| `exfull` / `exnone` | Target excised to an 11 Å halo around the interface |
| `tmfull` / `tmnone` | Single-chain self-template supplied instead of an MSA |
| `exrand*` / `exdiv*` | Depth-matched MSA ladder — random vs. diverse subsets at depth 2/5/10/30 |

## Contents

| Path | What it is |
| --- | --- |
| `code/` | The full pipeline — input construction, AF3 orchestration, scoring, analysis. See [`code/README.md`](code/README.md) for the file-by-file map and the run order. |
| `all_af3_folds.csv` | Scored results for all 26,241 folds (12 MB) |
| `SC4001.pdf` | The report |

## The results table

`all_af3_folds.csv` is one row per AF3 fold, keyed by
`(experiment, binder_id, target_id, condition, seed)`:

- **Scores** — `ipsae_min`, `ipsae_max`, `iptm`, `ptm`, `ranking_score`,
  `chain_pair_iptm_01` / `_10`, `fraction_disordered`, `has_clash`
- **Label** — `label` is 1 for a confirmed binder, 0 otherwise
  (3,146 / 26,241 rows; the positive rate per *design* is 10.7%)
- **Provenance** — `fold_dir`, `msa_unpaired_depth`, `msa_paired_depth`,
  `n_templates`, `crop_len`, `binder_seq`, `receptor_seq`, and `error` for any
  fold that failed

Coverage: 13 targets, 3 seeds (seed 1 everywhere; seeds 2 and 3 for the
`e2` arm), 10,677 `full` and 10,596 `none` folds.

```python
import pandas as pd
df = pd.read_csv("all_af3_folds.csv")
e2 = df[(df.experiment == "e2_out") & (df.seed == 1)]
print(e2.groupby("condition").ipsae_max.mean())
```

The analysis in the report is produced from this table by
`code/analyze_e2.py`; `python analyze_e2.py --selftest` validates the statistics
(AP, ROC-AUC, target-level paired bootstrap) before any of it is believed.

## Reproducing

Everything downstream of the folds runs on CPU with `numpy` / `pandas` /
`matplotlib`. Regenerating the folds themselves needs an NVIDIA GPU of compute
capability ≥ 8.0, the AF3 v3.0 sequence databases and the access-controlled AF3
model parameters:

```bash
rsync -av --exclude '.git' Project/ user@gpubox:~/Project/   # move it across
cd ~/Project/code

./setup.sh              # analysis deps, AF3 @ 4f41476, build
./fetch_databases.sh    # ~150 GB via aria2 (~300 GB unpacked)
./preflight.sh          # GPU capability, HMMER, weights, databases, deps
./reproduce_folds.sh    # all 26,241 folds, in nine resumable stages
```

Nothing hardcodes a path or a GPU count — `env.sh` resolves paths relative to
itself and detects GPUs with `nvidia-smi`, so the directory can be copied to
any Linux box with a compute-capability-8.0+ card and run as-is. Per-machine
overrides go in `code/local.env` (`cp local.env.example local.env`). Every
stage is resumable, and `DRY_RUN=1` prints the commands without running them.

The model parameters are access-controlled and must be requested and copied in
by hand; `preflight.sh` fails loudly if they are absent. The per-stage
breakdown, the GPU notes and the pitfalls worth knowing are in
[`code/README.md`](code/README.md).

Benchmark labels and the reference AF3 folds come from the
[Zenodo deposit 15722219](https://zenodo.org/records/15722219).
