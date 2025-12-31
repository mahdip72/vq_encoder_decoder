# Data Pipeline

This section covers the data pipeline for preprocessing experimental PDB/CIF files to remove noise, 
handle missing residues and chains, and produce a unified HDF5 format for high-throughput training and inference.

## Evaluation Datasets

| Dataset | Description | Download Link |
|---------|-------------|---------------|
| CAMEO2024 | CAMEO 2024 evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/ErhhREP9bH5AoBBOe5IshCUBix3KAvYvZpAS7f1FS3pB_g?e=gQPDWl) |
| CASP14 | CASP 14 evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EgMgJtM0fdNHpU46opUf0OgBZxhlJiV8Xu8N1Ke2lgw0mg?e=0d46eL) |
| CASP15 | CASP 15 evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EgMgJtM0fdNHpU46opUf0OgBZxhlJiV8Xu8N1Ke2lgw0mg?e=0d46eL) |
| CASP16 | CASP 16 evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EgMgJtM0fdNHpU46opUf0OgBZxhlJiV8Xu8N1Ke2lgw0mg?e=0d46eL) |
| Zero-Shot | Zero-shot evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EiPEh9RGgypEi_LRWlNhLi0BSlbFsr9VryhKT1v8MYLj7Q?e=Uhr3bF) |

## Download PDBs with Foldcomp (recommended)
We provide a helper script to fetch a Foldcomp-formatted database and extract structures to uncompressed `.pdb` files. See the official docs for more details: [Foldcomp README](https://github.com/steineggerlab/foldcomp) and the [Foldcomp download server](https://foldcomp.steineggerlab.workers.dev/).

Quick start (preferred):
```bash
# 1) Open the script and set parameters at the top:
#    - DATABASE_NAME (e.g. afdb_swissprot_v4, afdb_uniprot_v4, afdb_rep_v4, afdb_rep_dark_v4,
#      esmatlas, esmatlas_v2023_02, highquality_clust30, or organism sets like h_sapiens)
#    - DOWNLOAD_DIR (where DB files live)
#    - OUTPUT_DIR (where .pdb files will be written)

nano data/download_foldcomp_db_to_pdb.sh

# 2) Run the script
bash data/download_foldcomp_db_to_pdb.sh

# The script will (a) fetch the DB via the optional Python helper if available,
# or instruct you to download DB files from the Foldcomp server, then (b) call
# `foldcomp decompress` to write uncompressed .pdb files to OUTPUT_DIR.
```

Notes:
- You need the `foldcomp` CLI in your PATH. Install guidance is available in the [Foldcomp README](https://github.com/steineggerlab/foldcomp).
- The script optionally uses the Python package `foldcomp` to auto-download DB files. If not present, it prints the exact files to fetch from the official server.
- After PDBs are downloaded, continue with the converters below to produce the `.h5` dataset used by this repo.

## HDF5 format used by this repo
- **seq**: length-L amino-acid string. Standard 20-letter alphabet; **X** marks unknowns and numbering gaps.
- **N_CA_C_O_coord**: float array of shape (L, 4, 3). Backbone atom coordinates in Å for [N, CA, C, O] per residue. Missing atoms/residues are NaN-filled.
- **plddt_scores**: float array of shape (L,). Per-residue pLDDT pulled from B-factors when present; NaN if unavailable.

## Convert PDB/CIF → HDF5
This script scans a directory recursively and writes one `.h5` per processed chain.
- **Input format**: By default it searches for `.pdb`. Use `--use_cif` to read `.cif` files (no `.cif.gz`).
- **Chain filtering**: drops chains whose final length (after gap handling) is < `--min_len` or > `--max_len`.
- **Duplicate sequences**: among highly similar chains (identity > 0.95), keeps the one with the most resolved CA atoms.
- **Numbering gaps & insertions**: handles insertion codes natively. For numeric residue-number gaps (both PDB and CIF), inserts `X` residues with NaN coords. If a gap exceeds `--gap_threshold` (default 5), reduces the number of inserted residues using the straight-line CA-CA distance (assumes ~3.8 Å per residue); if CA coords are missing, caps at the threshold. This prevents runaway padding for CIF files with non-contiguous author numbering.
- **Outputs**: by default filenames are `<index>_<basename>.h5` or `<index>_<basename>_chain_id_<ID>.h5` for multi-chain structures. Add `--no_file_index` to omit the `<index>_` prefix.

Examples:
```bash
# Default: PDB input
python data/pdb_to_h5.py \
  --data /abs/path/to/pdb_root \
  --save_path /abs/path/to/output_h5 \
  --max_len 2048 \
  --min_len 25 \
  --max_workers 16
```

```bash
# CIF input (no .gz)
python data/pdb_to_h5.py \
  --use_cif \
  --data /abs/path/to/cif_root \
  --save_path /abs/path/to/output_h5
```

```bash
# Control large numeric gaps with CA-CA estimate (applies to PDB and CIF)
python data/pdb_to_h5.py \
  --data /abs/path/to/structures \
  --save_path /abs/path/to/output_h5 \
  --gap_threshold 5
```

```bash
# Omit index from output filenames
python data/pdb_to_h5.py \
  --no_file_index \
  --data /abs/path/to/pdb_or_cif_root \
  --save_path /abs/path/to/output_h5
```

## Convert HDF5 → PDB
Converts `.h5` backbones to PDB, writing only N/CA/C atoms and skipping residues with any NaN coordinates.

Example:
```bash
python data/h5_to_pdb.py \
  --h5_dir /abs/path/to/input_h5 \
  --pdb_dir /abs/path/to/output_pdb
```

## Split complexes into monomer PDBs
Scans a directory recursively and writes one PDB per selected chain, deduplicating highly similar chains.

- **Input format**: By default it searches for `.pdb`. Use `--use_cif` to read `.cif` files (no `.cif.gz`).
- **Chain filtering**: drops chains whose final length (after gap checks) is < `--min_len` or > `--max_len`.
- **Duplicate sequences**: among highly similar chains (identity > 0.90), keeps the one with the most resolved CA atoms.
- **Numbering gaps**: for large numeric residue-numbering gaps, uses the straight-line CA-CA distance to cap the number of inserted missing residues (quality control; outputs remain original coordinates).
- **Outputs**: default filenames are `<basename>_chain_id_<ID>.pdb`. Add `--with_file_index` to prefix with `<index>_`. Output chain ID is set to "A".

Examples:
```bash
# Default: PDB input
python data/break_complex_to_monumers.py \
  --data /abs/path/to/structures \
  --save_path /abs/path/to/output_pdb \
  --max_len 2048 \
  --min_len 25 \
  --max_workers 16
```

```bash
# CIF input (no .gz)
python data/break_complex_to_monumers.py \
  --use_cif \
  --data /abs/path/to/cif_root \
  --save_path /abs/path/to/output_pdb
```

## How inference/evaluation use `.h5`
- **Inference**: `inference_encode.py` and `inference_embed.py` read datasets from `.h5` in the format above. `inference_decode.py` decodes VQ indices (from CSV) to backbone coordinates; you can convert decoded `.h5`/coords to PDB with `data/h5_to_pdb.py`.
- **Evaluation**: `evaluation.py` consumes an `.h5` file via `data_path` in `configs/evaluation_config.yaml` and reports TM-score/RMSD; it can also write aligned PDBs.
