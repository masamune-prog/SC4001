#!/usr/bin/env bash
# Inference sweep, sharded across whatever GPUs this machine has.
#
#   ./run_e2.sh e2_in_full e2_out_full
#   ./run_e2.sh e2_in_none e2_out_none
#   ./run_e2.sh                      # both, in sequence (7,064 folds)
#
# Inputs already carry their alignments, so the data pipeline is OFF here --
# that is the whole point of running it once per target in run_msa_queue.sh.
# Any fold that already has a _confidences.json is skipped, so an interrupted
# run resumes by re-invoking the same script.
#
# GPUs come from env.sh, which detects them with nvidia-smi: one card or eight,
# no configuration. FOLDS_PER_GPU>1 puts several workers on each card, which is
# worth doing here because the folds are small (871 tokens at the largest) and
# a single one leaves a 48 GB card mostly idle.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$WORK_DIR"

shard() {               # shard <in_dir> <out_dir> <gpu> <index> <n_shards>
  local in_dir="$1" out_dir="$2" gpu="$3" idx="$4" n="$5"
  local i=0
  for json in "$in_dir"/*.json; do
    if (( i % n == idx )); then
      local name; name="$(basename "$json" .json)"
      if ls "$out_dir/$name"/*_confidences.json >/dev/null 2>&1; then
        i=$((i+1)); continue
      fi
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$AF3_PYTHON" "$AF3_REPO/run_alphafold.py" \
        --json_path="$json" \
        --output_dir="$out_dir" \
        --model_dir="$MODEL_DIR" \
        --db_dir="$DB_DIR" \
        --run_data_pipeline=false \
        --run_inference=true \
        >>"$out_dir/gpu${gpu}.w${idx}.log" 2>&1 || echo "[e2] FAILED $name" >&2
    fi
    i=$((i+1))
  done
}

sweep() {               # sweep <in_dir> <out_dir>
  local in_dir="$1" out_dir="$2"
  [[ -d "$in_dir" ]] || { echo "[e2] no such input dir: $in_dir" >&2; return 1; }
  mkdir -p "$out_dir"
  local total; total=$(ls "$in_dir"/*.json 2>/dev/null | wc -l)
  (( total > 0 )) || { echo "[e2] no inputs in $in_dir" >&2; return 1; }

  # One worker per (gpu, slot). Workers stride through the input list, so the
  # split stays even however many there are.
  IFS=',' read -r -a gpu_arr <<< "$GPUS"
  local workers=() g
  for (( slot=0; slot<FOLDS_PER_GPU; slot++ )); do
    for g in "${gpu_arr[@]}"; do workers+=("$g"); done
  done
  local n=${#workers[@]}
  echo "[e2] $in_dir -> $out_dir ($total folds, $n worker(s) over GPUs $GPUS)"

  for idx in "${!workers[@]}"; do
    shard "$in_dir" "$out_dir" "${workers[$idx]}" "$idx" "$n" &
  done
  wait
  local done_n
  done_n=$(find "$out_dir" -name '*_confidences.json' ! -name '*summary*' | wc -l)
  echo "[e2] $out_dir: $done_n / $total complete"
}

if (( $# >= 2 )); then
  sweep "$1" "$2"
else
  sweep e2_in_full e2_out_full
  sweep e2_in_none e2_out_none
fi
