#!/usr/bin/env bash
# AF3 data pipeline over the unique target sequences (~1.3 h at MSA_JOBS=3).
#
# Only the data pipeline is run (--run_inference=false): the product we want is
# the *_data.json holding unpairedMsa/pairedMsa for each unique target
# sequence.  One alignment per target sequence, not per binder -- 16 jobs, not
# 3,532.
#
# This stage is HMMER on CPU, not GPU: MSA_JOBS is bounded by cores and RAM
# (jackhmmer against BFD/MGnify is memory-hungry), not by the number of cards.
#
# HMMER 3.4; AF3 v3.0 sequence databases, protein subset only (~300 GB).  Note
# that run_alphafold.py validates every database path at startup, so the unused
# nucleic-acid databases and the mmCIF template store need zero-byte
# placeholders -- fetch_databases.sh writes them.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$WORK_DIR"

IN_DIR="${IN_DIR:-targets}"
OUT_DIR="${OUT_DIR:-targets}"

mkdir -p "$OUT_DIR"

run_one() {
  local json="$1"
  local name
  name="$(basename "$json" .json)"
  if ls "$OUT_DIR/$name"/*_data.json >/dev/null 2>&1; then
    echo "[msa] skip $name (already has a data json)"
    return 0
  fi
  echo "[msa] $name"
  "$AF3_PYTHON" "$AF3_REPO/run_alphafold.py" \
    --json_path="$json" \
    --output_dir="$OUT_DIR" \
    --model_dir="$MODEL_DIR" \
    --db_dir="$DB_DIR" \
    --run_data_pipeline=true \
    --run_inference=false \
    >"$OUT_DIR/$name.msa.log" 2>&1
}
export -f run_one
export AF3_REPO AF3_PYTHON DB_DIR MODEL_DIR OUT_DIR

ls "$IN_DIR"/*.json >/dev/null 2>&1 || {
  echo "[msa] no inputs in $IN_DIR -- run prepare_targets.py first" >&2; exit 1; }

ls "$IN_DIR"/*.json | xargs -P "$MSA_JOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "[msa] done -> $OUT_DIR/<target_uid>/*_data.json"
