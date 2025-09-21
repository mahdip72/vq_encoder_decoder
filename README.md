# GCP-VQVAE: A Geometry-Complete Language for Protein 3D Structure

<p align="center">
  <img src="src/logo.png" alt="GCP-VQVAE" width="70%" />
</p>


## Abstract

Converting protein tertiary structure into discrete tokens via vector-quantized variational autoencoders (VQVAEs) creates a language of 3D geometry and provides a natural interface between sequence and structure models. While pose invariance is commonly enforced, retaining chirality and directional cues without sacrificing reconstruction accuracy remains challenging. In this paper, we introduce GCP-VQVAE, a geometry-complete tokenizer built around a strictly SE(3)-equivariant GCPNet encoder that preserves orientation and chirality of protein backbone. We vector-quantize pose-invariant readouts into a 4096-token vocabulary, and a transformer decoder maps tokens back to backbone coordinates via a 6D rotation head trained with SE(3)-invariant objectives. 

Building on these properties, we train GCP‑VQVAE on a corpus of 24 million monomer protein backbone structures gathered from the AlphaFold Protein Structure Database. On the CAMEO‑2024, CASP15, and CASP16 evaluation datasets, the model achieves backbone RMSDs of 0.4377 Å, 0.5293 Å, and 0.7576 Å, respectively, and achieves 100% codebook utilization on a held‑out validation set, substantially outperforming prior VQ‑VAE–based tokenizers and achieving state-of-the-art performance. Lastly, we elaborate on the various applications of this foundation-like model, such as protein structure compression and the integration of generative AI models.

## News
- 



## Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA-compatible GPU
- 16GB+ GPU memory recommended for training


## Installation

### Option 1: Using Pre-built Docker Images

For AMD64 systems:
```bash
docker pull mahdip72/vqvae3d:amd_v3
docker run --gpus all -it mahdip72/vqvae3d:amd_v6
```

For ARM64 systems:
```bash
docker pull mahdip72/vqvae3d:arm_v2
docker run --gpus all -it mahdip72/vqvae3d:arm_v2
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

### Option 3: Python Virtual Environment Setup

Create and activate a Python virtual environment:
```bash
python3 -m venv vqvae_env
source vqvae_env/bin/activate  # On Windows: vqvae_env\Scripts\activate
```

Make the installation script executable and run it:
```bash
chmod +x install.sh
bash install.sh
```

### Option 4: Conda Environment Setup

Create and activate a conda environment:
```bash
conda create --name vqvae python=3.10
conda activate vqvae
```

Make the installation script executable and run it:
```bash
chmod +x install_conda.sh
bash install_conda.sh
```

For GCPNet model dependencies:
```bash
bash install_conda_gcpnet.sh
```

## Data

### HDF5 format used by this repo
- **seq**: length‑L amino‑acid string. Standard 20‑letter alphabet; **X** marks unknowns and numbering gaps.
- **N_CA_C_O_coord**: float array of shape (L, 4, 3). Backbone atom coordinates in Å for [N, CA, C, O] per residue. Missing atoms/residues are NaN‑filled.
- **plddt_scores**: float array of shape (L,). Per‑residue pLDDT pulled from B‑factors when present; NaN if unavailable.

### Convert PDB/CIF → HDF5 (`data/pdb_to_h5.py`)
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

### Convert HDF5 → PDB (`data/h5_to_pdb.py`)
Converts `.h5` backbones to PDB, writing only N/CA/C atoms and skipping residues with any NaN coordinates.

Example:
```bash
python data/h5_to_pdb.py \
  --h5_dir /abs/path/to/input_h5 \
  --pdb_dir /abs/path/to/output_pdb
```

### Split complexes into monomer PDBs (`data/break_complex_to_monumers.py`)
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

We welcome independent validation of our ESM3 and FoldToken-4 conversion scripts to further confirm their correctness.

## Results

The table below reproduces Table 2 from the manuscript: reconstruction accuracy on community benchmarks. Metrics are backbone TM-score (↑) and RMSD in Å (↓).

<table>
  <thead>
    <tr>
      <th style="text-align:right;">Dataset</th>
      <th style="text-align:left;">Metric</th>
      <th>GCP-VQVAE (Ours)</th>
      <th>FoldToken-4</th>
      <th>ESM-3 VQVAE</th>
      <th>Structure Tokenizer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:right;" rowspan="2">CASP14</td>
      <td>TM-score</td>
      <td>0.9890</td>
      <td>0.5410</td>
      <td>0.5042</td>
      <td>0.3624</td>
    </tr>
    <tr>
      <td>RMSD</td>
      <td>0.5431</td>
      <td>8.9838</td>
      <td>10.4611</td>
      <td>10.5344</td>
    </tr>
    <tr>
      <td style="text-align:right;" rowspan="2">CASP15</td>
      <td>TM-score</td>
      <td>0.9884</td>
      <td>0.3289</td>
      <td>0.3206</td>
      <td>0.2329</td>
    </tr>
    <tr>
      <td>RMSD</td>
      <td>0.5293</td>
      <td>14.6702</td>
      <td>13.1877</td>
      <td>14.8956</td>
    </tr>
    <tr>
      <td style="text-align:right;" rowspan="2">CASP16</td>
      <td>TM-score</td>
      <td>0.9857</td>
      <td>0.8055</td>
      <td>0.7685</td>
      <td>0.6058</td>
    </tr>
    <tr>
      <td>RMSD</td>
      <td>0.7567</td>
      <td>5.5094</td>
      <td>8.2640</td>
      <td>8.7106</td>
    </tr>
    <tr>
      <td style="text-align:right;" rowspan="2">CAMEO-2024</td>
      <td>TM-score</td>
      <td>0.9918</td>
      <td>0.4784</td>
      <td>0.4633</td>
      <td>0.3575</td>
    </tr>
    <tr>
      <td>RMSD</td>
      <td>0.4377</td>
      <td>12.1089</td>
      <td>12.1138</td>
      <td>13.5360</td>
    </tr>
  </tbody>
</table>

Notes:
- FoldToken-4 uses a 256-size vocabulary; others use 4096.
- The Structure Tokenizer baseline supports only sequence lengths 50–512; out-of-range samples are excluded for that column only.
- Evaluation scripts for baselines were reproduced where public tooling was incomplete; see repository docs for details.

## 📜 Citation

If you use this code or the pretrained models, please cite the following paper:

```bibtex
will be added soon
```
