#!/usr/bin/env bash
# End-to-end reproduction of every AF3 fold behind the report.
#
#   ./reproduce_folds.sh              # all stages, in order (~24 h on 3 GPUs)
#   ./reproduce_folds.sh msa e2       # just those stages
#   ./reproduce_folds.sh --list       # what the stages are
#   DRY_RUN=1 ./reproduce_folds.sh    # print the commands, run nothing
#
# Stages are resumable and idempotent.  Every fold runner skips any fold that
# already has a _confidences.json, and each stage skips work whose output is
# already on disk, so an interrupted run is resumed by re-invoking the same
# command.  Nothing here trains anything: AF3 is frozen, and the only thing
# that varies between conditions is what the model is allowed to see.
#
# Prerequisites (see README.md):
#   - AF3 commit 4f41476, built with `uv sync` then `build_data`
#   - jax[cuda12]==0.10.2, HMMER 3.4, >=1 NVIDIA GPU of compute capability >=8.0
#   - the access-controlled AF3 model parameters in MODEL_DIR
#   - ./fetch_databases.sh to populate DB_DIR
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$WORK_DIR"

# Everything configurable lives in env.sh (and local.env); only the dry-run
# switch is local to this script.
DRY_RUN="${DRY_RUN:-0}"

STAGES=(data gate msa e2 seeds excised template ladder score)

usage() {
  cat <<'EOF'
stages:
  data      Zenodo deposit: benchmark CSV (82 MB) + reference AF3 folds (2.0 GB)
  gate      E0 validation gate -- recompute ipSAE from the deposit's own PAE
  msa       16 target alignments (AF3 data pipeline only)           ~1.3 h
  e2        intact-target ablation, seed 1: 7,064 folds             ~7 h
  seeds     the same ablation at seeds 2 and 3: 14,128 folds        ~11 h
  excised   11 A halo crop, full/none: 828 folds                    ~1 h
  template  single-chain self-template, full/none: 828 folds        ~1 h
  ladder    depth-matched random/diverse MSA ladder: 3,312 folds    ~3 h
  score     rescore every arm and run the Study A analysis
EOF
}

say() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M)" "$*"; }
run() {
  if [[ "$DRY_RUN" == "1" ]]; then printf '  + %s\n' "$*"; else "$@"; fi
}

# fetch <url> <dest> [md5]  -- aria2 if present, else curl.
#
# The md5 is not decoration.  "Is the file non-empty?" is not the same question
# as "is the file complete?": aria2 preallocates the full length up front, so an
# interrupted 2 GB download leaves a file of exactly the right size full of
# holes.  Without the checksum this function reports "have AF3_outputs.tar.zstd"
# and the stage proceeds to tar, which fails with a decoding error that says
# nothing about the real cause.  With it, an incomplete file is re-fetched
# (aria2 resumes in place) and a corrupt one is reported as corrupt.
fetch() {
  local url="$1" dest="$2" md5="${3:-}"
  if [[ -s "$dest" ]]; then
    if [[ -z "$md5" ]]; then
      echo "  have $dest"; return 0
    fi
    # A leftover .aria2 control file is itself proof of an unfinished transfer.
    if [[ ! -e "$dest.aria2" ]] && \
       [[ "$(md5sum "$dest" | cut -d' ' -f1)" == "$md5" ]]; then
      echo "  have $dest (md5 ok)"; return 0
    fi
    echo "  $dest is incomplete or corrupt -- resuming"
  fi
  if command -v aria2c >/dev/null 2>&1; then
    run aria2c "$url" --dir="$(dirname "$dest")" --out="$(basename "$dest")" \
      --max-connection-per-server=16 --split=16 --min-split-size=1M \
      --continue=true --auto-file-renaming=false --allow-overwrite=false \
      --max-tries=10 --retry-wait=15 --console-log-level=warn \
      ${md5:+--checksum=md5="$md5"}
  else
    echo "  (aria2c not found -- falling back to curl, this will be slower)"
    run curl -L --fail --retry 5 -C - -o "$dest" "$url"
  fi

  # Verify after the fact too: the curl path has no checksum support, and a
  # mismatch here means a bad file on disk, not a transient network error.
  if [[ -n "$md5" && "$DRY_RUN" != "1" ]]; then
    local got; got="$(md5sum "$dest" | cut -d' ' -f1)"
    if [[ "$got" != "$md5" ]]; then
      echo "  md5 mismatch for $dest: got $got, expected $md5" >&2
      echo "  delete it and re-run the 'data' stage" >&2
      exit 1
    fi
    echo "  md5 ok: $dest"
  fi
}

require_dataset() {
  [[ -s "$DATASET" ]] || { echo "missing $DATASET -- run the 'data' stage first" >&2; exit 1; }
}

# --- stages ------------------------------------------------------------------

stage_data() {
  say "deposit: benchmark labels and reference folds"
  # md5s as published in the Zenodo record's own file metadata
  # (https://zenodo.org/api/records/15722219).
  fetch "$ZENODO/final_dataset.csv/content"     "$DATASET" \
        "3a69ee9b0fecf53924a8c6479bac146e"
  fetch "$ZENODO/AF3_outputs.tar.zstd/content"  AF3_outputs.tar.zstd \
        "6430cbacd4ea54191e71583db5faa90d"
  if [[ ! -d AF3_outputs ]]; then
    say "unpacking the confidences jsons only (the structures are not needed)"
    run tar --zstd -xf AF3_outputs.tar.zstd \
      --wildcards --no-wildcards-match-slash 'AF3_outputs/*/*_confidences.json'
  fi
}

# E0: before spending a GPU-day, confirm our ipSAE reimplementation reproduces
# the deposit's published scores from its own raw PAE.  --gate exits non-zero
# if Spearman rho falls below 0.99, which is the point of running it first.
stage_gate() {
  require_dataset
  say "E0 gate: ipSAE recomputed from the deposit's PAE"
  run "$AF3_PYTHON" "$CODE_DIR/recompute_ipsae_from_folds.py" \
    --af3-dir . --dataset "$DATASET" --out features_folds.csv --gate
  run "$AF3_PYTHON" "$CODE_DIR/build_features_from_deposit.py" --dataset "$DATASET"
  run "$AF3_PYTHON" "$CODE_DIR/analyze_e2.py" --selftest
}

stage_msa() {
  require_dataset
  say "target alignments: 16 unique sequences, not 3,532 complexes"
  run "$AF3_PYTHON" "$CODE_DIR/prepare_targets.py" --dataset "$DATASET"   # -> targets/, e2_ids.txt
  run "$CODE_DIR"/run_msa_queue.sh                               # ~1.3 h, 3 at a time
  run "$AF3_PYTHON" "$CODE_DIR/compute_neff.py"                           # -> neff80.tsv
}

stage_e2() {
  require_dataset
  say "intact-target ablation, seed 1 (7,064 folds)"
  run "$AF3_PYTHON" "$CODE_DIR/build_inputs.py" --ids_file e2_ids.txt --condition full \
      --out_dir e2_in_full --dataset "$DATASET" --seed 1
  run "$AF3_PYTHON" "$CODE_DIR/build_inputs.py" --ids_file e2_ids.txt --condition none \
      --out_dir e2_in_none --dataset "$DATASET" --seed 1
  run "$CODE_DIR"/run_e2.sh e2_in_full e2_out_full
  run "$CODE_DIR"/run_e2.sh e2_in_none e2_out_none
  run "$AF3_PYTHON" "$CODE_DIR/score_e2.py" --dataset "$DATASET" --out e2_scores.tsv \
      --roots e2_out_full e2_out_none
}

stage_seeds() {
  require_dataset
  say "seeds $SEEDS (14,128 folds)"
  # The seed has to appear in the fold name -- build_inputs.py --seed puts it
  # there.  Without it, seeds 2 and 3 land in timestamp-disambiguated
  # directories that cannot be attributed to a seed; that cost a full rerun.
  run "$CODE_DIR"/run_e2_seeds.sh
}

# The excision, template and ladder arms all read interface geometry out of the
# seed-1 'full' folds, so stage_e2 has to have finished.  They run on the 414
# designs belonging to the four receptors that are genuinely croppable.
ensure_ex_ids() {
  [[ -s ex_ids.txt ]] && return 0
  say "ex_ids.txt: designs against $CROPPED"
  if [[ "$DRY_RUN" == "1" ]]; then printf '  + (write ex_ids.txt)\n'; return 0; fi
  "$AF3_PYTHON" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["CODE_DIR"])
from dataset import load_dataset
keep = set(os.environ["CROPPED"].split(","))
df = load_dataset(os.environ["DATASET"])
ids = df.loc[df.target.isin(keep), "design_id"].tolist()
missing = keep - set(df.target.unique())
if missing:
    raise SystemExit(f"receptors not in the benchmark: {sorted(missing)}")
with open("ex_ids.txt", "w") as fh:
    fh.write("\n".join(ids) + "\n")
print(f"[ex_ids] {len(ids)} designs across {len(keep)} receptors")
PY
}

require_baseline() {
  [[ -d e2_out_full ]] || {
    echo "e2_out_full/ missing -- the 'e2' stage must finish first" >&2; exit 1; }
}

stage_excised() {
  require_dataset; require_baseline; ensure_ex_ids
  say "epitope excision, 11 A halo (828 folds)"
  run "$AF3_PYTHON" "$CODE_DIR/build_excised.py" --ids_file ex_ids.txt --condition full \
      --out_dir ex_in_full --dataset "$DATASET"
  run "$AF3_PYTHON" "$CODE_DIR/build_excised.py" --ids_file ex_ids.txt --condition none \
      --out_dir ex_in_none --dataset "$DATASET"
  run "$CODE_DIR"/run_e2.sh ex_in_full ex_out_full
  run "$CODE_DIR"/run_e2.sh ex_in_none ex_out_none
}

stage_template() {
  require_dataset; require_baseline; ensure_ex_ids
  say "single-chain self-template (828 folds)"
  # build_template.py filters both _atom_site and _pdbx_poly_seq_scheme to the
  # target chain -- filtering only one makes AF3's template parser fail on a
  # cross-reference lookup -- and injects a revision date, because AF3
  # date-filters templates and a predicted model carries none.
  run "$AF3_PYTHON" "$CODE_DIR/build_template.py" --ids_file ex_ids.txt --condition full \
      --out_dir tm_in_full --dataset "$DATASET"
  run "$AF3_PYTHON" "$CODE_DIR/build_template.py" --ids_file ex_ids.txt --condition none \
      --out_dir tm_in_none --dataset "$DATASET"
  run "$CODE_DIR"/run_e2.sh tm_in_full tm_out_full
  run "$CODE_DIR"/run_e2.sh tm_in_none tm_out_none
}

stage_ladder() {
  require_dataset; require_baseline; ensure_ex_ids
  # Depth-matched: a random subset and a diversity-maximising subset at each
  # rung, so depth is held fixed and only the diversity of the alignment moves.
  # Reuses the excision windows, so stage_excised has to have run.
  [[ -s excision_windows.json || "$DRY_RUN" == "1" ]] || {
    echo "excision_windows.json missing -- run the 'excised' stage first" >&2; exit 1; }
  say "depth-matched MSA ladder (3,312 folds)"
  run "$AF3_PYTHON" "$CODE_DIR/build_ladder.py" --ids_file ex_ids.txt --out_dir e3_in --dataset "$DATASET"
  run "$CODE_DIR"/run_e2.sh e3_in e3_out
}

stage_score() {
  require_dataset
  say "rescoring every arm"
  roots=()
  for d in e2_out_full e2_out_none ex_out_full ex_out_none \
           tm_out_full tm_out_none e3_out; do
    [[ -d "$d" ]] && roots+=("$d")
  done
  for s in $SEEDS; do
    for c in full none; do
      [[ -d "e2_out_${c}_s${s}" ]] && roots+=("e2_out_${c}_s${s}")
    done
  done
  (( ${#roots[@]} > 0 )) || { echo "no fold directories found" >&2; exit 1; }
  run "$AF3_PYTHON" "$CODE_DIR/score_e2.py" --dataset "$DATASET" --out e2_scores.tsv --roots "${roots[@]}"
  say "Study A analysis"
  run "$AF3_PYTHON" "$CODE_DIR/analyze_e2.py" all --selftest --figures --cropped "$CROPPED"
  echo
  echo "  scores   e2_scores.tsv"
  echo "  analysis e2_analysis.json"
  echo "  figures  figs/"
}

# --- dispatch ----------------------------------------------------------------

if [[ "${1:-}" == "--list" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage; exit 0
fi

selected=("$@")
(( ${#selected[@]} == 0 )) && selected=("${STAGES[@]}")

for s in "${selected[@]}"; do
  if [[ ! " ${STAGES[*]} " == *" $s "* ]]; then
    echo "unknown stage '$s'" >&2; usage >&2; exit 2
  fi
done

say "configuration"
af3_show_config

started=$(date +%s)
for s in "${selected[@]}"; do
  "stage_$s"
done
say "done in $(( ($(date +%s) - started) / 60 )) min: ${selected[*]}"
