# GCP-VQVAE: A Geometry-Complete Language for Protein 3D Structure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![bioRxiv Preprint](https://img.shields.io/badge/bioRxiv-10.1101/2025.10.01.679833v1-brightgreen)](https://www.biorxiv.org/content/10.1101/2025.10.01.679833v2)

<p align="center">
  <img src="src/logo.png" alt="GCP-VQVAE" width="600" />
</p>


## Abstract

Converting protein tertiary structure into discrete tokens via vector-quantized variational autoencoders (VQ-VAEs) creates a language of 3D geometry and provides a natural interface between sequence and structure models. While pose invariance is commonly enforced, retaining chirality and directional cues without sacrificing reconstruction accuracy remains challenging. In this paper, we introduce GCP-VQVAE, a geometry-complete tokenizer built around a strictly SE(3)-equivariant GCPNet encoder that preserves orientation and chirality of protein backbones. We vector-quantize rotation/translation-invariant readouts that retain chirality into a 4096-token vocabulary, and a transformer decoder maps tokens back to backbone coordinates via a 6D rotation head trained with SE(3)-invariant objectives.

Building on these properties, we train GCP-VQVAE on a corpus of 24 million monomer protein backbone structures gathered from the AlphaFold Protein Structure Database. On the CAMEO2024, CASP15, and CASP16 evaluation datasets, the model achieves backbone RMSDs of 0.4377 Å, 0.5293 Å, and 0.7567 Å, respectively, and achieves 100% codebook utilization on a held-out validation set, substantially outperforming prior VQ-VAE–based tokenizers and achieving state-of-the-art performance. Beyond these benchmarks, on a zero-shot set of 1938 completely new experimental structures, GCP-VQVAE attains a backbone RMSD of 0.8193 Å and a TM-score of 0.9673, demonstrating robust generalization to unseen proteins. Lastly, we elaborate on the various applications of this foundation-like model, such as protein structure compression and the integration of generative protein language models. We make the GCP-VQVAE source code, zero-shot dataset, and its pretrained weights fully open for the research community.

<img src="src/reconstruction.png" alt="Reconstruction Example" width="1000">

## News
- 🗓️ **25 Sept 2025** — 🎉 Our paper was accepted to the NeurIPS 2025 AI4Science workshop!
- 🗓️ **3 Oct 2025** — Our preprint has been published in [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.10.01.679833v2).
- 🗓️ **10 Oct 2025** — 🚀 Pretrained checkpoints and evaluation datasets are now available for download!
- 🗓️ **17 Oct 2025** - Release a lite version of GCP-VQVAE with half the parameters.



## Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA-compatible GPU
- 16GB+ GPU memory recommended for training


## Installation

### Option 1: Using Pre-built Docker Images

For AMD64 systems:
```bash
docker pull mahdip72/vqvae3d:amd_v8
docker run --gpus all -it mahdip72/vqvae3d:amd_v8
```

For ARM64 systems:
```bash
docker pull mahdip72/vqvae3d:arm_v3
docker run --gpus all -it mahdip72/vqvae3d:arm_v3
```

### Option 2: Building from Dockerfile

```bash
# Clone the repository
git clone https://github.com/mahdip72/vq_encoder_decoder.git
cd vq_encoder_decoder

# Build the Docker image
docker build -t vqvae3d .

# Run the container
docker run --gpus all -it vqvae3d
```

#### (Optional, Hopper only) FlashAttention-3
If you are on H100, H800, GH200, H200 (SM90) you can enable FlashAttention-3 for faster, lower‑memory attention.

Build with FA3 baked in:
```bash
docker build --build-arg FA3=1 -t vqvae3d-fa3 .
```

## Data

### Evaluation Datasets

| Dataset | Description | Download Link |
|---------|-------------|---------------|
| CAMEO2024 | CAMEO 2024 evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/ErhhREP9bH5AoBBOe5IshCUBix3KAvYvZpAS7f1FS3pB_g?e=gQPDWl) |
| CASP14 | CASP 14 evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EgMgJtM0fdNHpU46opUf0OgBZxhlJiV8Xu8N1Ke2lgw0mg?e=0d46eL) |
| CASP15 | CASP 15 evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EgMgJtM0fdNHpU46opUf0OgBZxhlJiV8Xu8N1Ke2lgw0mg?e=0d46eL) |
| CASP16 | CASP 16 evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EgMgJtM0fdNHpU46opUf0OgBZxhlJiV8Xu8N1Ke2lgw0mg?e=0d46eL) |
| Zero-Shot | Zero-shot evaluation dataset | [Download](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mpngf_umsystem_edu/EiPEh9RGgypEi_LRWlNhLi0BSlbFsr9VryhKT1v8MYLj7Q?e=Uhr3bF) |

### Download PDBs with Foldcomp (recommended)
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

### HDF5 format used by this repo
- **seq**: length‑L amino‑acid string. Standard 20‑letter alphabet; **X** marks unknowns and numbering gaps.
- **N_CA_C_O_coord**: float array of shape (L, 4, 3). Backbone atom coordinates in Å for [N, CA, C, O] per residue. Missing atoms/residues are NaN‑filled.
- **plddt_scores**: float array of shape (L,). Per‑residue pLDDT pulled from B‑factors when present; NaN if unavailable.

### Convert PDB/CIF → HDF5
This script scans a directory recursively and writes one `.h5` per processed chain.
- **Input format**: By default it searches for `.pdb`. Use `--use_cif` to read `.cif` files (no `.cif.gz`).
- **Chain filtering**: drops chains whose final length (after gap handling) is < `--min_len` or > `--max_len`.
- **Duplicate sequences**: among highly similar chains (identity > 0.95), keeps the one with the most resolved CA atoms.
- **Numbering gaps & insertions**: handles insertion codes natively. For numeric residue‑number gaps (both PDB and CIF), inserts `X` residues with NaN coords. If a gap exceeds `--gap_threshold` (default 5), reduces the number of inserted residues using the straight‑line CA–CA distance (assumes ~3.8 Å per residue); if CA coords are missing, caps at the threshold. This prevents runaway padding for CIF files with non‑contiguous author numbering.
- **Outputs**: by default filenames are `<index>_<basename>.h5` or `<index>_<basename>_chain_id_<ID>.h5` for multi‑chain structures. Add `--no_file_index` to omit the `<index>_` prefix.

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
# Control large numeric gaps with CA–CA estimate (applies to PDB and CIF)
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

### Convert HDF5 → PDB
Converts `.h5` backbones to PDB, writing only N/CA/C atoms and skipping residues with any NaN coordinates.

Example:
```bash
python data/h5_to_pdb.py \
  --h5_dir /abs/path/to/input_h5 \
  --pdb_dir /abs/path/to/output_pdb
```

### Split complexes into monomer PDBs
Scans a directory recursively and writes one PDB per selected chain, deduplicating highly similar chains.

- **Input format**: By default it searches for `.pdb`. Use `--use_cif` to read `.cif` files (no `.cif.gz`).
- **Chain filtering**: drops chains whose final length (after gap checks) is < `--min_len` or > `--max_len`.
- **Duplicate sequences**: among highly similar chains (identity > 0.90), keeps the one with the most resolved CA atoms.
- **Numbering gaps**: for large numeric residue‑numbering gaps, uses the straight‑line CA–CA distance to cap the number of inserted missing residues (quality control; outputs remain original coordinates).
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

### How inference/evaluation use `.h5`
- **Inference**: `inference_encode.py` and `inference_embed.py` read datasets from `.h5` in the format above. `inference_decode.py` decodes VQ indices (from CSV) to backbone coordinates; you can convert decoded `.h5`/coords to PDB with `data/h5_to_pdb.py`.
- **Evaluation**: `evaluation.py` consumes an `.h5` file via `data_path` in `configs/evaluation_config.yaml` and reports TM‑score/RMSD; it can also write aligned PDBs.

## Usage

Before you begin:
- Prepare your dataset in `.h5` format as described in [Data](#data). Use the PDB → HDF5 converter in `data/pdb_to_h5.py`.

### Training

Configure your training parameters in `configs/config_vqvae.yaml` and run:

Note:
- Training expects datasets in the HDF5 layout defined in [HDF5 format used by this repo](#hdf5-format-used-by-this-repo).

```bash
# Set up accelerator configuration for multi-GPU training
accelerate config

# Start training with accelerate for multi-GPU support
accelerate launch train.py --config_path configs/config_vqvae.yaml
```

See the [Accelerate documentation](https://huggingface.co/docs/accelerate/index) for more options and configurations.

### Inference

### Pretrained Models

| Model | Description | Download Link                                                                                                                                |
|-------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Large | Full GCP-VQVAE model with best performance | [Download](https://mailmissouri-my.sharepoint.com/:u:/g/personal/mpngf_umsystem_edu/EaxLj74pK5BArOpPkF9MkDgBHxlfaDpAElPRiwH9BsIedA?e=34ida8) |
| Lite | Lightweight version for faster inference | [Download](https://mailmissouri-my.sharepoint.com/:u:/g/personal/mpngf_umsystem_edu/EUXZF_Is2X9IrjLeIWk7T5gBNb3yliRwVOAWi2rHyyympg?e=VAveKw) |                                                                                                                                |

**Setup Instructions:**
1. Download the zip file of the checkpoint
2. Extract the checkpoint folder
3. Set the `trained_model_dir` path in your config file (following ones) to point to the right checkpoint.


Multi‑GPU with Hugging Face Accelerate:
- The following scripts support multi‑GPU via Accelerate: `inference_encode.py`, `inference_embed.py`, `inference_decode.py`, and `evaluation.py`.

Example (2 GPUs, bfloat16):
```bash
accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=2 evaluation.py
```
Or like in Training, configure Accelerate first:
```bash
accelerate config
accelerate launch evaluation.py
```


See the [Accelerate documentation](https://huggingface.co/docs/accelerate/index) for more options and configurations.

All inference scripts consume `.h5` inputs in the format defined in [Data](#data).

To extract the VQ codebook embeddings:
```bash
python codebook_extraction.py
```
Edit `configs/inference_codebook_extraction_config.yaml` to change paths and output filename.

To encode proteins into discrete VQ indices:
```bash
python inference_encode.py
```
Edit `configs/inference_encode_config.yaml` to change dataset paths, model, and output. Input datasets should be `.h5` as in [HDF5 format used by this repo](#hdf5-format-used-by-this-repo).

To extract per‑residue embeddings from the VQ layer:
```bash
python inference_embed.py
```
Edit `configs/inference_embed_config.yaml` to change dataset paths, model, and output HDF5. Input `.h5` files must follow [HDF5 format used by this repo](#hdf5-format-used-by-this-repo).

To decode VQ indices back to 3D backbone structures:
```bash
python inference_decode.py
```
Edit `configs/inference_decode_config.yaml` to point to the indices CSV and adjust runtime. To write PDBs from decoded outputs, see [Convert HDF5 → PDB](#convert-hdf5--pdb-datah5_to_pdbpy).

### Evaluation

To evaluate predictions and write TM‑score/RMSD along with aligned PDBs:
```bash
python evaluation.py
```

Notes:
- Set `data_path` to an `.h5` dataset that follows [HDF5 format used by this repo](#hdf5-format-used-by-this-repo).
- To visualize results as PDB, convert `.h5` outputs with [`data/h5_to_pdb.py`](#convert-hdf5--pdb-datah5_to_pdbpy).

Example config template (`configs/evaluation_config.yaml`):
```yaml
trained_model_dir: "/abs/path/to/trained_model"   # Folder containing checkpoint and saved YAMLs
checkpoint_path: "checkpoints/best_valid.pth"     # Relative to trained_model_dir
config_vqvae: "config_vqvae.yaml"                 # Names of saved training YAMLs
config_encoder: "config_gcpnet_encoder.yaml"
config_decoder: "config_geometric_decoder.yaml"

data_path: "/abs/path/to/evaluation/data.h5"      # HDF5 used for evaluation
output_base_dir: "evaluation_results"              # A timestamped subdir is created inside

batch_size: 8
shuffle: true
num_workers: 0
max_task_samples: 5000000                           # Optional cap
vq_indices_csv_filename: "vq_indices.csv"          # Also writes observed VQ indices
alignment_strategy: "kabsch"                       # "kabsch" or "no"
mixed_precision: "bf16"                            # "no", "fp16", "bf16", "fp8"

tqdm_progress_bar: true
```

## External Tokenizer Evaluations

We evaluated additional VQ-VAE backbones alongside GCP-VQVAE:

- ESM3 VQVAE (forked repo: [mahdip72/esm](https://github.com/mahdip72/esm)) – community can reuse `pdb_to_tokens.py` and `tokens_to_pdb.py` that we authored because the upstream project lacks ready-to-use scripts.
- FoldToken-4 (forked repo: [mahdip72/FoldToken_open](https://github.com/mahdip72/FoldToken_open)) – we rewrote `foldtoken/pdb_to_token.py` and `foldtoken/token_to_pdb.py` for better performance and efficiency with negligible increase in error.
- Structure Tokenizer ([instadeepai/protein-structure-tokenizer](https://github.com/instadeepai/protein-structure-tokenizer)) – results reproduced with the official implementation.

Let us know if you need the scripts of more baseline tokenizers evaluated in the recent update of the manuscript.

## Results

The table below reproduces Table 2 from the manuscript: reconstruction accuracy on community benchmarks and a zero-shot setting. Metrics are backbone TM-score (↑) and RMSD in Å (↓).

<p align="center">
  <img src="src/results.png" alt="Benchmark results across datasets" width="1000">
</p>

## Experimental Features

- Added an experimental option to compress token sequences using latent codebooks inspired by [ByteDance’s 1D tokenizer](https://github.com/bytedance/1d-tokenizer) (TikTok); this enables configurable compression factors within our VQ pipeline.
- Added the residual quantization (multi-depth VQ) using a shared codebook when `tik_tok.residual_depth > 1`. This only works when tik_tok is enabled which means the residual latents are packed depth-by-depth, flattened into a single stream for NTP and decoding, and their masks/embeddings remain aligned with the flattened indices. This improves reconstruction capacity without expanding the base codebook.
- Included an optional next-token prediction head, drawing on the autoregressive regularization ideas from *“When Worse is Better: Navigating the Compression-Generation Tradeoff in Visual Tokenization”*, to encourage codebooks that are friendlier to autoregressive modeling.
- Enabled adaptive loss coefficients driven by gradient norms: each active loss (MSE, distance/direction, VQ, NTP) tracks its synchronized gradient magnitude and scales its weight toward the 0.2–5.0 norm “comfort zone.” Coefficients shrink when a loss overpowers the rest and grow when its gradients fade, keeping the multi-objective training balanced without constant manual re-tuning.

## Codebook Usage Statistics

Enable `model.vqvae.vector_quantization.log_codebook_usage_statistics: true` to log the following statistics to TensorBoard under `codebook_usage_statistics/*` during validation:

| Metric | TensorBoard tag | Range |
| --- | --- | --- |
| Unigram entropy (bits) | `codebook_usage_statistics/entropy_unigram_bits` | 0 → log₂(K) (12 bits for K=4096) |
| Unigram perplexity | `codebook_usage_statistics/perplexity_unigram` | 1 → K |
| Bigram conditional entropy H₂\|₁ | `codebook_usage_statistics/entropy_bigram_cond_bits` | 0 → log₂(K) |
| Bigram perplexity | `codebook_usage_statistics/perplexity_bigram` | 1 → K |
| Trigram conditional entropy H₃\|₂₁ | `codebook_usage_statistics/entropy_trigram_cond_bits` | 0 → log₂(K) |
| Trigram perplexity | `codebook_usage_statistics/perplexity_trigram` | 1 → K |
| ΔH₁ = H₁ − H₂\|₁ | `codebook_usage_statistics/delta_entropy_h1_h2` | 0 → H₁ |
| ΔH₂ = H₁ − H₃\|₂₁ | `codebook_usage_statistics/delta_entropy_h1_h3` | 0 → H₁ |
| Extra conditional gain H₂\|₁ − H₃\|₂₁ | `codebook_usage_statistics/delta_entropy_conditional` | 0 → H₂\|₁ |
| Mutual information (lag d) | `codebook_usage_statistics/mutual_info_lag{d}` | ≥0, typically ≤2 bits |
| Zipf slope | `codebook_usage_statistics/zipf_slope` | negative (≈−0.5 to −1.2) |
| Zipf R² | `codebook_usage_statistics/zipf_r2` | 0 → 1 |
| Active codes | `codebook_usage_statistics/active_codes` | 0 → K |
| Effective usage ratio (PPL₁/active) | `codebook_usage_statistics/effective_usage_ratio` | 0 → 1 |


## Acknowledgments

This repository builds upon several excellent open-source projects:

- [**vector-quantize-pytorch**](https://github.com/lucidrains/vector-quantize-pytorch) – Vector quantization implementations used in our VQ-VAE architecture.
- [**x-transformers**](https://github.com/lucidrains/x-transformers) – Transformer components integrated into our encoder and decoder modules of VQ-VAE.
- [**ProteinWorkshop**](https://github.com/a-r-j/ProteinWorkshop) – We slimmed the original workshop down to the GCPNet core and extended it with:
  - opt-in `torch.compile` support that now powers our training, inference, and evaluation paths (configurable in every config file).
  - faster aggregation backends: a CSR/`torch.segment_reduce` implementation for PyTorch-only setups and an automatic switch to cuGraph-ops fused aggregators when `pylibcugraphops` is installed.
  - multi-GPU friendly evaluation utilities that reuse the same compilation flags as training, keeping the code path consistent across scripts.

We gratefully acknowledge **NVIDIA** for providing computational resources via their dedicated server that made 
this project feasible, enabling us to train and evaluate our models at scale.


## 📜 Citation

If you use this code or the pretrained models, please cite the following paper:

```bibtex
@article{Pourmirzaei2025gcpvqvae,
  author  = {Pourmirzaei, Mahdi and Morehead, Alex and Esmaili, Farzaneh and Ren, Jarett and Pourmirzaei, Mohammadreza and Xu, Dong},
  title   = {GCP-VQVAE: A Geometry-Complete Language for Protein 3D Structure},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.10.01.679833},
  url     = {https://www.biorxiv.org/content/10.1101/2025.10.01.679833v1}
}
```
