#!/usr/bin/env bash

#
# Download a Foldcomp database and extract entries to uncompressed PDB files.
#
# How to use:
#   - Set DATABASE_NAME and OUTPUT_DIR below.
#   - Run this script (no CLI args needed).
#   - The script will download the Foldcomp DB files if missing and
#     then decompress them to PDB files in OUTPUT_DIR.
#
# Requirements:
#   - foldcomp CLI available in PATH (https://github.com/steineggerlab/foldcomp)
#   - python3 with "foldcomp" package for convenient DB download helper
#     (optional if you place DB files manually in DOWNLOAD_DIR)
#
# Notes:
#   - Supported database names are listed in Foldcomp README, e.g.:
#       afdb_uniprot_v4, afdb_swissprot_v4, afdb_rep_v4, afdb_rep_dark_v4,
#       esmatlas, esmatlas_v2023_02, highquality_clust30, and specific organisms.
#   - The script is idempotent: it will skip downloading if files already exist.
#

set -euo pipefail
IFS=$'\n\t'

# ------------------------
# Parameters (edit here)
# ------------------------

# Example: "afdb_swissprot_v4" or "afdb_uniprot_v4" or "esmatlas"
DATABASE_NAME="afdb_swissprot_v4"

# Where to store/download the foldcomp DB files (index, lookup, dbtype, source)
DOWNLOAD_DIR="/mnt/hdd8/mehdi/projects/vq_encoder_decoder/data/foldcomp_dbs"

# Where to write uncompressed .pdb outputs
OUTPUT_DIR="/mnt/hdd8/mehdi/projects/vq_encoder_decoder/data/pdb_from_foldcomp/${DATABASE_NAME}"

# Number of threads for decompression
THREADS="8"

# If you already have the DB files somewhere else, set this to that directory and
# the script will skip Python download and just use foldcomp to decompress.
# For auto-download via Python helper, leave as-is.

# ------------------------
# End of parameters
# ------------------------

echo "[Info] DATABASE_NAME       : ${DATABASE_NAME}"
echo "[Info] DOWNLOAD_DIR        : ${DOWNLOAD_DIR}"
echo "[Info] OUTPUT_DIR          : ${OUTPUT_DIR}"
echo "[Info] THREADS             : ${THREADS}"

command -v foldcomp >/dev/null 2>&1 || {
  echo "[Error] foldcomp CLI not found in PATH. Please install it: https://github.com/steineggerlab/foldcomp" >&2
  exit 1
}

mkdir -p "${DOWNLOAD_DIR}" "${OUTPUT_DIR}"

DB_PREFIX_PATH="${DOWNLOAD_DIR}/${DATABASE_NAME}"
DB_MAIN_FILE="${DB_PREFIX_PATH}"
DB_INDEX_FILE="${DB_PREFIX_PATH}.index"
DB_DBTYPE_FILE="${DB_PREFIX_PATH}.dbtype"
DB_LOOKUP_FILE="${DB_PREFIX_PATH}.lookup"

have_all_db_files() {
  [[ -s "${DB_MAIN_FILE}" && -s "${DB_INDEX_FILE}" && -s "${DB_DBTYPE_FILE}" && -s "${DB_LOOKUP_FILE}" ]]
}

echo "[Step] Ensuring Foldcomp database files are present..."
if have_all_db_files; then
  echo "[OK] Found existing DB files for ${DATABASE_NAME} in ${DOWNLOAD_DIR}"
else
  echo "[Info] DB files not found. Attempting Python helper download via 'foldcomp' package..."
  if command -v python3 >/dev/null 2>&1; then
    # Try to import foldcomp and download
    set +e
    python3 - <<PYCODE
try:
    import foldcomp, os
    target = os.path.abspath('${DOWNLOAD_DIR}')
    os.makedirs(target, exist_ok=True)
    # foldcomp.setup downloads into CWD; change dir temporarily
    prev = os.getcwd()
    os.chdir(target)
    foldcomp.setup('${DATABASE_NAME}')
    os.chdir(prev)
    print('PY_OK')
except Exception as e:
    print('PY_ERR:', e)
PYCODE
    py_status=$?
    set -e
    if [[ ${py_status} -ne 0 ]]; then
      echo "[Warn] Python attempt failed (non-zero exit)."
    fi
  else
    echo "[Warn] python3 not found; skipping Python-based download."
  fi

  if have_all_db_files; then
    echo "[OK] DB files downloaded successfully."
  else
    echo "[Error] DB files still missing."
    echo "        You can download manually from the Foldcomp download server and place files here:"
    echo "        https://foldcomp.steineggerlab.workers.dev/"
    echo "        Required files: ${DATABASE_NAME}, ${DATABASE_NAME}.index, ${DATABASE_NAME}.dbtype, ${DATABASE_NAME}.lookup"
    exit 1
  fi
fi

echo "[Step] Decompressing '${DATABASE_NAME}' to PDB into: ${OUTPUT_DIR}"

# foldcomp decompress [-t number] <dir|tar(.gz)|db> [<dir|tar>]
foldcomp decompress -t "${THREADS}" "${DB_MAIN_FILE}" "${OUTPUT_DIR}"

echo "[Done] Decompression complete. PDB files are in: ${OUTPUT_DIR}"

echo "[Tip] To subset large DBs by IDs, see Foldcomp README and mmseqs createsubdb."


