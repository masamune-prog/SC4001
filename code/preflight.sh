#!/usr/bin/env bash
# Does this machine have everything the pipeline needs?
#
#   ./preflight.sh            # check and report
#   ./preflight.sh --quiet    # exit status only
#
# Run this straight after copying the project to a new machine.  It checks the
# things that otherwise fail hours into a sweep: a GPU too old for AF3, model
# parameters that were never copied across, a database directory missing one
# file, an interpreter that cannot import AF3.
#
# Exit: 0 everything ready, 1 something required is missing, 2 warnings only.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

QUIET=0; [[ "${1:-}" == "--quiet" ]] && QUIET=1
fail=0; warn=0

say()  { (( QUIET )) || printf '%s\n' "$*"; }
ok()   { (( QUIET )) || printf '  \033[32mok\033[0m    %s\n' "$*"; }
bad()  { (( QUIET )) || printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
note() { (( QUIET )) || printf '  \033[33mwarn\033[0m  %s\n' "$*"; warn=1; }

say
say "configuration"
(( QUIET )) || af3_show_config

# --- host --------------------------------------------------------------------
say
say "host"
if [[ "$(uname -s)" == "Linux" ]]; then
  ok "Linux ($(uname -r))"
else
  bad "$(uname -s) -- AF3 only supports Linux. Analysis scripts still run here."
fi

cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "?")
ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
if (( ram_gb >= 64 )); then ok "RAM ${ram_gb} GB, ${cores} cores"
elif (( ram_gb > 0 )); then note "RAM ${ram_gb} GB -- AF3 recommends >=64 GB for the data pipeline"
else note "could not read /proc/meminfo"; fi

# --- GPU ---------------------------------------------------------------------
say
say "gpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  n_gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  cap=$(af3_compute_cap)
  (( QUIET )) || nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
       --format=csv,noheader 2>/dev/null | sed 's/^/        gpu /'
  if (( cap >= 80 )); then
    ok "$n_gpu GPU(s), compute capability $((cap/10)).$((cap%10)) (AF3 needs >= 8.0)"
  elif (( cap > 0 )); then
    bad "compute capability $((cap/10)).$((cap%10)) -- AF3 requires >= 8.0"
  else
    note "could not read compute capability"
  fi
  # The largest fold here is 871 tokens; 48 GB cards have room to spare.
  minmem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
           | sort -n | head -1)
  if [[ -n "$minmem" ]]; then
    if (( minmem >= 24000 )); then
      ok "smallest card ${minmem} MiB (largest fold in this project is 871 tokens)"
    else
      note "smallest card ${minmem} MiB -- fine for 871 tokens, but keep FOLDS_PER_GPU=1"
    fi
  fi
  (( QUIET )) || echo "        using GPUS=$GPUS at FOLDS_PER_GPU=$FOLDS_PER_GPU"
else
  bad "nvidia-smi not found -- no GPU, folding cannot run (analysis still can)"
fi

# --- tools -------------------------------------------------------------------
say
say "tools"
for t in jackhmmer hmmsearch hmmbuild nhmmer; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t"; else bad "$t not found (HMMER 3.4)"; fi
done
for t in zstd aria2c; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t"; else note "$t not found (needed by fetch_databases.sh)"; fi
done

# --- python ------------------------------------------------------------------
say
say "python"
pyver=$("$AF3_PYTHON" -c 'import sys; print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")
if [[ -n "$pyver" ]]; then ok "$AF3_PYTHON is Python $pyver"
else bad "AF3_PYTHON ($AF3_PYTHON) is not runnable"; fi

for m in numpy pandas; do
  if "$AF3_PYTHON" -c "import $m" 2>/dev/null; then
    ok "$m $("$AF3_PYTHON" -c "import $m;print($m.__version__)" 2>/dev/null)"
  else bad "$m missing (pip install -r requirements.txt)"; fi
done
if "$AF3_PYTHON" -c "import matplotlib" 2>/dev/null; then ok "matplotlib"
else note "matplotlib missing -- only needed for analyze_e2.py --figures"; fi

if "$AF3_PYTHON" -c "import jax" 2>/dev/null; then
  jaxdev=$("$AF3_PYTHON" -c "import jax;print(','.join(d.platform for d in jax.devices()))" 2>/dev/null)
  if [[ "$jaxdev" == *gpu* || "$jaxdev" == *cuda* ]]; then ok "jax sees: $jaxdev"
  else bad "jax is installed but sees no GPU (devices: ${jaxdev:-none})"; fi
else note "jax not importable by $AF3_PYTHON -- set AF3_PYTHON to the AF3 venv"; fi

if "$AF3_PYTHON" -c "import alphafold3" 2>/dev/null; then
  ok "alphafold3 importable"
else
  note "alphafold3 not importable by $AF3_PYTHON (ok if you run it another way)"
fi

# --- AF3 installation --------------------------------------------------------
say
say "alphafold3"
if [[ -f "$AF3_REPO/run_alphafold.py" ]]; then ok "$AF3_REPO/run_alphafold.py"
else bad "run_alphafold.py not found under AF3_REPO=$AF3_REPO"; fi

if [[ -d "$MODEL_DIR" ]] && compgen -G "$MODEL_DIR/*" >/dev/null; then
  ok "model parameters in $MODEL_DIR ($(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1))"
else
  bad "no model parameters in MODEL_DIR=$MODEL_DIR (these are access-controlled; request and copy them)"
fi

# --- databases ---------------------------------------------------------------
say
say "databases"
if [[ -d "$DB_DIR" ]]; then
  if "$CODE_DIR/fetch_databases.sh" --check >/dev/null 2>&1; then
    ok "$DB_DIR complete"
  else
    bad "$DB_DIR incomplete -- run ./fetch_databases.sh (see ./fetch_databases.sh --check)"
  fi
else
  bad "DB_DIR=$DB_DIR does not exist -- run ./fetch_databases.sh"
fi

# --- project data ------------------------------------------------------------
say
say "project"
if [[ -s "$DATASET" ]]; then ok "dataset $DATASET ($(du -h "$DATASET" | cut -f1))"
else note "dataset not present -- ./reproduce_folds.sh data will fetch it"; fi

if [[ -d "$WORK_DIR" ]]; then
  avail=$(df -Pk "$WORK_DIR" | awk 'NR==2 {print int($4/1048576)}')
  # A full sweep is ~26k fold directories; budget ~1 GB per 1k folds of json.
  if (( avail >= 100 )); then ok "work dir $WORK_DIR, ${avail} GB free"
  else note "work dir $WORK_DIR has only ${avail} GB free -- a full sweep writes 26k fold dirs"; fi
else
  note "WORK_DIR=$WORK_DIR does not exist yet"
fi

if "$AF3_PYTHON" -c "import sys; sys.path.insert(0,'$CODE_DIR'); import metrics; metrics.selftest()" >/dev/null 2>&1; then
  ok "metrics selftest passes"
else
  note "metrics selftest did not pass (needs numpy/pandas; run: python analyze_e2.py --selftest)"
fi

# --- verdict -----------------------------------------------------------------
say
if (( fail )); then
  say "NOT READY -- fix the FAIL lines above."
  exit 1
elif (( warn )); then
  say "ready, with warnings."
  exit 2
else
  say "ready."
  exit 0
fi
