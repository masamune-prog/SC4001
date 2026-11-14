#!/usr/bin/env bash
# AF3 v3.0 sequence databases, protein subset only, fetched with aria2.
#
#   ./fetch_databases.sh              # download, decompress, write placeholders
#   DB_DIR=/mnt/ssd/af3 ./fetch_databases.sh
#   ./fetch_databases.sh --check      # verify an existing DB_DIR, download nothing
#
# DB_DIR comes from env.sh (default $HOME/af3/databases); set it in local.env
# to put the databases on a fast disk.
#
# Why aria2 and not curl: these are five files totalling ~150 GB compressed.
# aria2c opens CONNS parallel ranged connections per file, which on a fast link
# is the difference between a couple of hours and most of a day.  It also
# resumes a partial file in place (--continue), so an interrupted download is
# restarted by re-running this script.
#
# Only the protein databases are fetched.  The pipeline folds protein-protein
# complexes and never touches the nucleic-acid databases or the mmCIF template
# store -- but run_alphafold.py stats every database path at startup and exits
# if one is missing, so those get zero-byte placeholders instead of 700 GB of
# downloads we would never read.
#
# Requires: aria2c, zstd.  (apt install aria2 zstd)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

BASE="${BASE:-https://storage.googleapis.com/alphafold-databases/v3.0}"
CONNS="${CONNS:-16}"          # parallel connections per file
JOBS="${JOBS:-2}"             # files downloaded concurrently
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"   # 1 keeps the .zst after decompression

# Protein subset.  ~150 GB compressed, ~300 GB on disk after decompression.
PROTEIN_DBS=(
  bfd-first_non_consensus_sequences.fasta        #   9 GB -> jackhmmer BFD
  mgy_clusters_2022_05.fa                        #  65 GB -> MGnify
  uniprot_all_2021_04.fa                         #  45 GB -> paired MSA
  uniref90_2022_05.fa                            #  31 GB -> unpaired MSA
  pdb_seqres_2022_09_28.fasta                    #  <1 GB -> template search
)

# Never read by a protein-protein fold; present only to satisfy startup checks.
PLACEHOLDER_FILES=(
  rnacentral_active_seq_id_90_cov_80_linclust.fasta
  nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta
  rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta
)
PLACEHOLDER_DIRS=(
  mmcif_files
)

have() { command -v "$1" >/dev/null 2>&1; }

check() {
  local missing=0
  for f in "${PROTEIN_DBS[@]}"; do
    if [[ -s "$DB_DIR/$f" ]]; then
      printf '  ok      %-55s %s\n' "$f" "$(du -h "$DB_DIR/$f" | cut -f1)"
    else
      printf '  MISSING %-55s\n' "$f"; missing=1
    fi
  done
  for f in "${PLACEHOLDER_FILES[@]}"; do
    [[ -e "$DB_DIR/$f" ]] || { printf '  MISSING %-55s (placeholder)\n' "$f"; missing=1; }
  done
  for d in "${PLACEHOLDER_DIRS[@]}"; do
    [[ -d "$DB_DIR/$d" ]] || { printf '  MISSING %-55s (placeholder dir)\n' "$d"; missing=1; }
  done
  return "$missing"
}

if [[ "${1:-}" == "--check" ]]; then
  echo "[db] checking $DB_DIR"
  check && echo "[db] complete" || { echo "[db] incomplete"; exit 1; }
  exit 0
fi

have aria2c || { echo "aria2c not found (apt install aria2)" >&2; exit 1; }
have zstd   || { echo "zstd not found (apt install zstd)" >&2; exit 1; }

mkdir -p "$DB_DIR"

# Free space check before committing to a multi-hour download.  ~300 GB
# decompressed, plus the compressed copy alongside it while unpacking.
avail_gb=$(df -Pk "$DB_DIR" | awk 'NR==2 {print int($4/1048576)}')
need_gb=$(( KEEP_ARCHIVES == 1 ? 450 : 320 ))
echo "[db] $DB_DIR: ${avail_gb} GB free, ~${need_gb} GB needed"
if (( avail_gb < need_gb )); then
  echo "[db] not enough free space" >&2
  exit 1
fi

# --- download ----------------------------------------------------------------
# One aria2c invocation over a generated input list: -j downloads JOBS files at
# a time, -x/-s give each file CONNS ranged connections.  --continue resumes a
# partial file; --auto-file-renaming=false makes a re-run overwrite-resume
# rather than writing name.1, name.2 ...
todo=()
for f in "${PROTEIN_DBS[@]}"; do
  if [[ -s "$DB_DIR/$f" ]]; then
    echo "[db] skip $f (already decompressed)"
  else
    todo+=("$f")
  fi
done

if (( ${#todo[@]} > 0 )); then
  list="$(mktemp)"
  trap 'rm -f "$list"' EXIT
  for f in "${todo[@]}"; do
    printf '%s/%s.zst\n  dir=%s\n  out=%s.zst\n' "$BASE" "$f" "$DB_DIR" "$f" >>"$list"
  done

  echo "[db] downloading ${#todo[@]} file(s), ${JOBS} at a time, ${CONNS} connections each"
  aria2c \
    --input-file="$list" \
    --max-concurrent-downloads="$JOBS" \
    --max-connection-per-server="$CONNS" \
    --split="$CONNS" \
    --min-split-size=1M \
    --continue=true \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --file-allocation=falloc \
    --max-tries=10 \
    --retry-wait=15 \
    --summary-interval=30 \
    --console-log-level=warn

  # --- decompress -------------------------------------------------------------
  # Serial: zstd -d is already multi-threaded and these writes are disk-bound.
  for f in "${todo[@]}"; do
    echo "[db] decompressing $f"
    if (( KEEP_ARCHIVES == 1 )); then
      zstd --decompress --force --threads=0 "$DB_DIR/$f.zst" -o "$DB_DIR/$f"
    else
      zstd --decompress --force --rm --threads=0 "$DB_DIR/$f.zst" -o "$DB_DIR/$f"
    fi
  done
fi

# --- placeholders ------------------------------------------------------------
for f in "${PLACEHOLDER_FILES[@]}"; do
  [[ -e "$DB_DIR/$f" ]] || { : >"$DB_DIR/$f"; echo "[db] placeholder $f"; }
done
for d in "${PLACEHOLDER_DIRS[@]}"; do
  [[ -d "$DB_DIR/$d" ]] || { mkdir -p "$DB_DIR/$d"; echo "[db] placeholder dir $d/"; }
done

echo
echo "[db] contents of $DB_DIR"
check
echo "[db] done.  Export DB_DIR=$DB_DIR for the fold scripts."
