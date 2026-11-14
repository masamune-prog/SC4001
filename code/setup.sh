#!/usr/bin/env bash
# One-time setup on a fresh machine.
#
#   ./setup.sh              # analysis deps + AF3 checkout and build
#   ./setup.sh --analysis   # analysis deps only (no GPU, no AF3)
#
# Installs into $AF3_HOME (default $HOME/af3) so nothing large lands inside the
# project directory.  Does not fetch databases (./fetch_databases.sh) and
# cannot fetch the model parameters -- those are access-controlled and must be
# requested from DeepMind, then copied into $MODEL_DIR.
#
# Afterwards:  ./preflight.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

AF3_COMMIT="${AF3_COMMIT:-4f41476}"   # the commit the report was produced with
ANALYSIS_ONLY=0; [[ "${1:-}" == "--analysis" ]] && ANALYSIS_ONLY=1

say() { printf '\n\033[1m[setup] %s\033[0m\n' "$*"; }

say "configuration"
af3_show_config

# --- analysis side -----------------------------------------------------------
# numpy/pandas/matplotlib only; the re-ranker that needed torch is not in this
# reconstruction.
say "analysis dependencies"
"$AF3_PYTHON" -m pip install --upgrade pip
"$AF3_PYTHON" -m pip install -r "$CODE_DIR/requirements.txt"

if (( ANALYSIS_ONLY )); then
  say "analysis-only setup done. ./analyze_e2.py --selftest to verify."
  exit 0
fi

# --- system packages ---------------------------------------------------------
say "system tools"
missing=()
for t in jackhmmer zstd aria2c git; do
  command -v "$t" >/dev/null 2>&1 || missing+=("$t")
done
if (( ${#missing[@]} > 0 )); then
  echo "  missing: ${missing[*]}"
  echo "  on Debian/Ubuntu:  sudo apt install -y hmmer zstd aria2 git"
  echo "  (hmmer provides jackhmmer; AF3 wants HMMER 3.4)"
  echo
  read -r -p "  continue anyway? [y/N] " yn
  [[ "$yn" == [yY]* ]] || exit 1
else
  echo "  all present"
fi

# --- AF3 ---------------------------------------------------------------------
say "alphafold3 @ $AF3_COMMIT -> $AF3_REPO"
mkdir -p "$AF3_HOME" "$MODEL_DIR"
if [[ -d "$AF3_REPO/.git" ]]; then
  echo "  already cloned; fetching"
  git -C "$AF3_REPO" fetch --all --tags
else
  git clone https://github.com/google-deepmind/alphafold3.git "$AF3_REPO"
fi
git -C "$AF3_REPO" checkout "$AF3_COMMIT"

say "building alphafold3"
if command -v uv >/dev/null 2>&1; then
  ( cd "$AF3_REPO" && uv sync && uv run build_data )
  echo
  echo "  built into $AF3_REPO/.venv"
  echo "  set AF3_PYTHON=$AF3_REPO/.venv/bin/python in local.env"
else
  echo "  uv not found. Either install it:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  or install AF3 into the current interpreter:"
  echo "    cd $AF3_REPO && pip install -e . && build_data"
  echo "    pip install 'jax[cuda12]==0.10.2'"
  exit 1
fi

# --- next steps --------------------------------------------------------------
say "remaining steps"
cat <<EOF
  1. Model parameters are access-controlled and cannot be scripted. Request
     them from DeepMind, then put them in:
         $MODEL_DIR
  2. Databases (~150 GB download, ~300 GB on disk):
         ./fetch_databases.sh
  3. Verify everything:
         ./preflight.sh
  4. Run:
         ./reproduce_folds.sh --list
EOF
