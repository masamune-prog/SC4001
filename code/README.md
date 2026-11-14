# Does AlphaFold 3 Need Evolution to Recognise a Binder? — code

Reconstruction of the pipeline described in appendix A/D of the report, after
the original working copy was lost. AF3 is frozen throughout; nothing here
trains or fine-tunes it.

## What is and is not here

| File | Purpose |
| --- | --- |
| `dataset.py` | Benchmark loader + the 3,766 → 3,669 → 3,532 retention funnel |
| `msa_utils.py` | a3m parsing, Neff80, column cropping, depth-matched subsampling |
| `cif_utils.py` | Minimal mmCIF reader/writer; CA/heavy-atom coordinates |
| `af3_io.py` | AF3 input JSON construction, fold naming, output discovery |
| `metrics.py` | AP, ROC-AUC, tie-breaking, target-level paired bootstrap, selftest |
| `ipsae_recompute.py` | ipSAE from an AF3 confidences JSON (equation 1) |
| `prepare_targets.py` | Unique target sequences → data-pipeline inputs |
| `compute_neff.py` | Neff80 at 80% identity → `neff80.tsv` |
| `build_inputs.py` | Intact-target conditions (`full`, `none`); `--seed` sets the fold name |
| `build_excised.py` | 11 Å halo crop + matched alignment column crop |
| `build_template.py` | Single-chain mmCIF self-template |
| `build_ladder.py` | Depth-matched random and diverse subsets |
| `score_e2.py` | Scoring → `e2_scores.tsv` |
| `analyze_e2.py` | Paired bootstrap, class decomposition, every Study A table/figure |
| `run_msa_queue.sh`, `run_e2.sh`, `run_e2_seeds.sh` | GPU orchestration |
| `env.sh`, `local.env.example` | Shared config; GPU detection; per-machine overrides |
| `setup.sh` | One-time install on a new machine (deps, AF3 checkout, build) |
| `preflight.sh` | Is this machine ready? GPU, HMMER, weights, databases, deps |
| `fetch_databases.sh` | AF3 v3.0 protein databases via aria2, + placeholders |
| `reproduce_folds.sh` | Every fold end to end, in nine resumable stages |
| `recompute_ipsae_from_folds.py` | ipSAE from the deposit's raw PAE; the E0 gate |
| `build_features_from_deposit.py` | The 16 design-side features, z-scored |

**Not reproduced, by request:** `train_reranker.py` and `eval_reranker.py` —
the Study B MLP and its evaluation. The two feature-table builders above are
kept because the E0 validation gate depends on them.

## Dependencies

Analysis side: Python 3.12, `numpy`, `pandas`, `matplotlib` (figures only).
No PyTorch — that was only the re-ranker.

AF3 side (not runnable on this machine): Linux, NVIDIA GPU of compute
capability ≥ 8.0, AF3 commit `4f41476` built with `uv sync` then `build_data`,
`jax[cuda12]==0.10.2`, HMMER 3.4, AF3 v3.0 sequence databases (protein subset,
~300 GB), and the access-controlled model parameters.

`run_alphafold.py` validates every database path at startup, so the unused
nucleic-acid databases and the mmCIF template store need zero-byte
placeholders.

## Moving this to a GPU machine

Copy the directory across and run four commands. Nothing in the scripts
hardcodes a path or a GPU count: `env.sh` resolves every path relative to its
own location and detects GPUs with `nvidia-smi`, so one card or eight both work
unconfigured.

```bash
rsync -av --exclude '.git' Project/ user@gpubox:~/Project/
ssh user@gpubox
cd ~/Project/code

./setup.sh          # analysis deps, AF3 @ 4f41476, uv sync + build_data
./fetch_databases.sh    # ~150 GB via aria2, unpacked to ~300 GB
./preflight.sh      # GPU capability, HMMER, weights, databases, deps
./reproduce_folds.sh
```

`setup.sh` cannot fetch the model parameters — they are access-controlled, so
request them from DeepMind and copy them into `$MODEL_DIR` yourself.
`preflight.sh` fails loudly if they are missing, which is the point: it checks
the things that otherwise surface hours into a sweep.

To override anything per machine, `cp local.env.example local.env` and edit it;
`local.env` is gitignored, so each machine keeps its own paths. The defaults
put AF3, its weights and its databases under `$HOME/af3` rather than inside the
project, because the project directory is OneDrive-synced and the databases are
300 GB. Set `WORK_DIR` in `local.env` to send fold outputs to scratch rather
than into the synced tree — a full sweep writes ~26k directories.

### On an RTX 6000 Ada / A6000

Both are compute capability 8.9 and 8.6 respectively, over AF3's floor of 8.0,
with 48 GB each. The largest fold in this project is 871 tokens (median 166)
against AF3's documented 5,120-token limit on an 80 GB card, so unified memory
is not needed and `preflight.sh` will say so. What 48 GB does buy is
concurrency: set `FOLDS_PER_GPU=2` (or more) in `local.env` and `run_e2.sh`
puts that many workers on each card, since one small fold leaves it mostly
idle.

Expect throughput below A100 — these cards have roughly half the memory
bandwidth, and AF3 inference is largely bandwidth-bound — so the stage timings
below will stretch. Time a hundred folds before committing to the full sweep.
One caveat for the report: AF3's own docs state that numerical accuracy was
validated on A100 and H100 only, other devices being "believed to be accurate"
but untested at scale. The E0 gate is unaffected (it recomputes ipSAE from the
deposit's stored PAE, never touching the GPU), and the ablation is paired
within-condition on the same card, so this is a Limitations sentence rather
than a threat to the result.

## Pipeline

Two scripts drive everything; `reproduce_folds.sh` is the pipeline below with
the ordering constraints and the resume logic already encoded.

```bash
./fetch_databases.sh             # ~150 GB via aria2, then unpack
./fetch_databases.sh --check     # verify without downloading
./preflight.sh                   # is this machine ready?

./reproduce_folds.sh --list      # the nine stages and what each costs
./reproduce_folds.sh             # all of them, in order (~24 h on 3 GPUs)
./reproduce_folds.sh e2 score    # or a subset
DRY_RUN=1 ./reproduce_folds.sh   # print the commands, run nothing
```

Configuration comes from `env.sh`, overridable per machine in `local.env` and
per invocation by environment variable: `AF3_REPO`, `MODEL_DIR`, `DB_DIR`,
`AF3_PYTHON`, `DATASET`, `WORK_DIR`, `GPUS`, `FOLDS_PER_GPU`, `MSA_JOBS`,
`SEEDS`, `CROPPED`. Every stage is resumable: fold runners skip any fold that already has a
`_confidences.json`, and each stage skips work whose output is on disk, so an
interrupted run is resumed by re-invoking the same command.

The individual steps, if you would rather drive them by hand:

```bash
# benchmark labels and reference scores (82 MB)
curl -L -o final_dataset.csv \
  https://zenodo.org/api/records/15722219/files/final_dataset.csv/content
# the deposit's own AF3 folds, for the E0 gate (2.0 GB)
curl -L -o AF3_outputs.tar.zstd \
  https://zenodo.org/api/records/15722219/files/AF3_outputs.tar.zstd/content

python prepare_targets.py            # 16 target sequences -> targets/
./run_msa_queue.sh                   # AF3 data pipeline, 3 at a time (~1.3 h)
python compute_neff.py               # -> neff80.tsv (30 s)

python build_inputs.py --ids_file e2_ids.txt --condition full --out_dir e2_in_full
python build_inputs.py --ids_file e2_ids.txt --condition none --out_dir e2_in_none
./run_e2.sh                          # 7,064 folds, 3 GPUs (~7 h)
python score_e2.py                   # -> e2_scores.tsv (30 s)
python analyze_e2.py --selftest      # validates the statistics
./run_e2_seeds.sh                    # seeds 2 and 3, 14,128 folds (~11 h)

# single-seed arms (each needs the seed-1 'full' folds for its geometry)
python build_excised.py  --ids_file ex_ids.txt --condition full --out_dir ex_in_full
python build_excised.py  --ids_file ex_ids.txt --condition none --out_dir ex_in_none
python build_template.py --ids_file ex_ids.txt --condition full --out_dir tm_in_full
python build_template.py --ids_file ex_ids.txt --condition none --out_dir tm_in_none
python build_ladder.py   --out_dir e3_in     # 3,312 ladder inputs
./run_e2.sh ex_in_full ex_out_full           # ...and so on per arm

python score_e2.py --roots e2_out_* ex_out_* tm_out_* e3_out_* --out e2_scores.tsv
python analyze_e2.py all --figures --cropped IL7Ra,SARS_CoV2_RBD,Pdl1,sntx
```

`ex_ids.txt` is not produced by any of the builders — it is the 414 designs
belonging to the four receptors that are genuinely croppable. The `excised`
stage of `reproduce_folds.sh` writes it from `CROPPED` if it is absent.

E0 gate, before any production run:

```bash
tar --zstd -xf AF3_outputs.tar.zstd \
  --wildcards --no-wildcards-match-slash 'AF3_outputs/*/*_confidences.json'
python recompute_ipsae_from_folds.py --af3-dir . \
  --dataset final_dataset.csv --out features_folds.csv --gate
```

## Databases

`fetch_databases.sh` pulls the five protein databases of AF3 v3.0 from
`storage.googleapis.com/alphafold-databases/v3.0` with `aria2c`, sixteen ranged
connections per file and two files at a time — on a fast link that is a couple
of hours rather than most of a day, and a partial file resumes in place.

| File | Compressed | Used for |
| --- | --- | --- |
| `bfd-first_non_consensus_sequences.fasta` | 9 GB | jackhmmer BFD |
| `mgy_clusters_2022_05.fa` | 65 GB | MGnify |
| `uniprot_all_2021_04.fa` | 45 GB | paired MSA |
| `uniref90_2022_05.fa` | 31 GB | unpaired MSA |
| `pdb_seqres_2022_09_28.fasta` | <1 GB | template search |

~300 GB once decompressed. The three nucleic-acid databases and the mmCIF
template store are never read by a protein-protein fold, so the script writes
zero-byte placeholders for them rather than fetching ~700 GB that would go
untouched — `run_alphafold.py` stats every database path at startup and exits
if one is missing.

## Things worth knowing before you run anything

- **The seed must be in the fold name.** `build_inputs.py --seed` puts it
  there. The earlier defect — seeds 2 and 3 landing in timestamp-disambiguated
  directories that could not be attributed to a seed — cost a full rerun.
- **Templates filter two categories, not one.** `build_template.py` filters
  both `_atom_site` and `_pdbx_poly_seq_scheme` to the target chain; filtering
  only one makes AF3's template parser fail on a cross-reference lookup. It
  also injects `_pdbx_audit_revision_history.revision_date`, because AF3
  date-filters templates and a predicted model carries no release date.
- **Order matters.** `build_excised.py` and `build_template.py` read geometry
  out of the seed-1 `full` folds, so `run_e2.sh` has to finish first.
- **Fold runners are resumable.** Any fold with a `_confidences.json` is
  skipped, so an interrupted sweep resumes by re-invoking the same script.

## Two places where the deposit's schema has to be checked

Both are column-name questions that cannot be settled without the CSV in hand:

1. `dataset.py: ALIASES` — the design id, target, sequence and label columns.
   The loader prints the funnel counts; they should read 3,766 → 3,669 → 3,532
   across 13 targets with 391 confirmed binders (10.7%). If they do not, an
   alias is wrong.
2. `build_features_from_deposit.py: INPUT_FEATURES` — the 16 design-side
   features. The script prints which ones it could not resolve.

The report also carries an open inconsistency worth settling here: Methods says
21 AF3 + 16 input = 37 features, while the old Limitations text said "5 to 33
AF3 features". `AF3_FEATURES` in `recompute_ipsae_from_folds.py` lists 21 and
prints how many were actually available in a given set of folds — use that
count, and make it consistent everywhere in the report.
