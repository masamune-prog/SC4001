#!/usr/bin/env bash
# Seeds beyond the seed-1 baseline (14,128 folds at SEEDS="2 3").
#
# The seed is part of the fold name (build_inputs.py --seed).  It must be:
# before that was corrected, seeds 2 and 3 wrote to timestamp-disambiguated
# directories that could not be attributed to a seed, and those folds had to be
# discarded and rerun.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$WORK_DIR"

for seed in $SEEDS; do
  for cond in full none; do
    in_dir="e2_in_${cond}_s${seed}"
    out_dir="e2_out_${cond}_s${seed}"
    "$AF3_PYTHON" "$CODE_DIR/build_inputs.py" --ids_file e2_ids.txt \
        --condition "$cond" --out_dir "$in_dir" --dataset "$DATASET" --seed "$seed"
    "$CODE_DIR/run_e2.sh" "$in_dir" "$out_dir"
  done
done

echo "[seeds] rescoring every arm"
"$AF3_PYTHON" "$CODE_DIR/score_e2.py" --dataset "$DATASET" --out e2_scores.tsv --roots \
  e2_out_full e2_out_none \
  $(for s in $SEEDS; do echo "e2_out_full_s$s e2_out_none_s$s"; done)
