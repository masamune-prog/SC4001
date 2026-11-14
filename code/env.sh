#!/usr/bin/env bash
# Shared configuration, sourced by every script in this directory.
#
# Nothing here needs editing to move the project to another machine: the
# defaults are either relative to this directory or detected at runtime.  To
# override any of it, copy local.env.example to local.env and edit that --
# local.env is gitignored, so each machine keeps its own paths.
#
# Precedence:  environment variable  >  local.env  >  the defaults below.
#
# Not executable on its own:  source env.sh

# --- where we are ------------------------------------------------------------
# Resolved from this file, so the scripts work from any working directory and
# survive the whole project being copied somewhere else.
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CODE_DIR/.." && pwd)"
export CODE_DIR PROJECT_ROOT

# --- machine-specific overrides ----------------------------------------------
if [[ -f "$CODE_DIR/local.env" ]]; then
  # shellcheck source=/dev/null
  set -a; source "$CODE_DIR/local.env"; set +a
fi

# --- AF3 installation --------------------------------------------------------
# AF3 itself, its weights and its databases are far too large to live inside a
# synced project directory, so they default to $HOME/af3.
: "${AF3_HOME:=$HOME/af3}"
: "${AF3_REPO:=$AF3_HOME/alphafold3}"
: "${MODEL_DIR:=$AF3_HOME/models}"
: "${DB_DIR:=$AF3_HOME/databases}"
# Interpreter used for everything. Prefers an AF3 venv if setup.sh made one,
# then whatever `python` resolves to, then python3.
if [[ -z "${AF3_PYTHON:-}" ]]; then
  if [[ -x "$AF3_REPO/.venv/bin/python" ]]; then AF3_PYTHON="$AF3_REPO/.venv/bin/python"
  elif command -v python >/dev/null 2>&1;  then AF3_PYTHON="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then AF3_PYTHON="$(command -v python3)"
  else AF3_PYTHON=python3
  fi
fi

# --- data --------------------------------------------------------------------
# Relative to the code directory, so a transferred copy finds its own files.
: "${DATASET:=$CODE_DIR/final_dataset.csv}"
: "${WORK_DIR:=$CODE_DIR}"          # where fold inputs/outputs are written

# --- run parameters ----------------------------------------------------------
: "${SEEDS:=2 3}"                                    # beyond the seed-1 baseline
: "${CROPPED:=IL7Ra,SARS_CoV2_RBD,Pdl1,sntx}"        # the four excision targets
: "${MSA_JOBS:=3}"                                   # concurrent HMMER jobs
: "${FOLDS_PER_GPU:=1}"                              # concurrent folds per GPU
: "${ZENODO:=https://zenodo.org/api/records/15722219/files}"

# --- GPUs --------------------------------------------------------------------
# Detected rather than assumed: a machine with one card, or four, works with no
# configuration.  Set GPUS=0,1 by hand to use a subset.
af3_detect_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo ""; return; fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | paste -sd, -
}

if [[ -z "${GPUS:-}" ]]; then
  GPUS="$(af3_detect_gpus)"
  [[ -z "$GPUS" ]] && GPUS=0      # no nvidia-smi: assume a single device
fi

# Compute capability of the first visible GPU, as an integer (86 for 8.6).
af3_compute_cap() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | tr -d ' .' | sed 's/[^0-9]//g'
}

# --- XLA / JAX ---------------------------------------------------------------
# PREALLOCATE=false lets several folds share a card; the triton_gemm flag is
# AF3's own default and avoids a long compile-time stall.
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
: "${XLA_FLAGS:=--xla_gpu_enable_triton_gemm=false}"

# Unified memory is only needed above ~5,120 tokens; the largest fold in this
# project is 871 (crop_len + binder_len), so it stays off unless asked for.
if [[ "${UNIFIED_MEMORY:-0}" == "1" ]]; then
  export TF_FORCE_UNIFIED_MEMORY=true
  : "${XLA_CLIENT_MEM_FRACTION:=3.2}"
  export XLA_CLIENT_MEM_FRACTION
fi

# Compute capability 7.x needs a pass disabled; 8.0+ does not.  Harmless to
# check here so a V100 box does not silently produce bad numbers.
if [[ "$(af3_compute_cap)" =~ ^7 ]]; then
  XLA_FLAGS="$XLA_FLAGS --xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi

export AF3_HOME AF3_REPO MODEL_DIR DB_DIR AF3_PYTHON
export DATASET WORK_DIR SEEDS CROPPED MSA_JOBS FOLDS_PER_GPU ZENODO GPUS
export XLA_PYTHON_CLIENT_PREALLOCATE XLA_FLAGS

af3_show_config() {
  cat <<EOF
  project      $PROJECT_ROOT
  af3 repo     $AF3_REPO
  model dir    $MODEL_DIR
  db dir       $DB_DIR
  dataset      $DATASET
  work dir     $WORK_DIR
  gpus         $GPUS  (${FOLDS_PER_GPU} fold(s) per gpu)
  seeds        1 $SEEDS
  cropped      $CROPPED
EOF
}
