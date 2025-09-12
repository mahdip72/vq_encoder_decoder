# GCP-VQVAE: A Geometry-Complete Language for Protein 3D Structure

## Abstract

Converting protein tertiary structure into discrete tokens via vector-quantized variational autoencoders (VQVAEs) creates a language of 3D geometry and provides a natural interface between sequence and structure models. While pose invariance is commonly enforced, retaining chirality and directional cues without sacrificing reconstruction accuracy remains challenging. In this paper, we introduce GCP-VQVAE, a geometry-complete tokenizer built around a strictly SE(3)-equivariant GCPNet encoder that preserves orientation and chirality of protein backbone. We vector-quantize pose-invariant readouts into a 4096-token vocabulary, and a transformer decoder maps tokens back to backbone coordinates via a 6D rotation head trained with SE(3)-invariant objectives. 

Building on these properties, we train GCP‑VQVAE on a corpus of 24 million monomer protein backbone structures gathered from the AlphaFold Protein Structure Database. On the CAMEO‑2024, CASP15, and CASP16 evaluation datasets, the model achieves backbone RMSDs of 0.4377\,Å, 0.5293\,Å, and 0.7576\,Å, respectively, and achieves 100\% codebook utilization on a held‑out validation set, substantially outperforming prior VQ‑VAE–based tokenizers and achieving state-of-the-art performance. Lastly, we elaborate on the various applications of this foundation-like model, such as protein structure compression and the integration of generative AI models.


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

## Usage

### Training

Configure your training parameters in `configs/config_vqvae.yaml` and run:

```bash
# Set up accelerator configuration for multi-GPU training
accelerate config

# Start training with accelerate for multi-GPU support
accelerate launch train.py --config_path configs/config_vqvae.yaml
```

### Inference

To extract the VQ codebook embeddings:
```bash
python codebook_extraction.py
```
Edit `configs/inference_codebook_extraction_config.yaml` to change paths and output filename.

To encode proteins into discrete VQ indices:
```bash
python inference_encode.py
```
Edit `configs/inference_encode_config.yaml` to change dataset paths, model, and output.

To extract per‑residue embeddings from the VQ layer:
```bash
python inference_embed.py
```
Edit `configs/inference_embed_config.yaml` to change dataset paths, model, and output HDF5.

To decode VQ indices back to 3D backbone structures:
```bash
python inference_decode.py
```
Edit `configs/inference_decode_config.yaml` to point to the indices CSV and adjust runtime.

### Evaluation

To evaluate predictions and write TM‑score/RMSD along with aligned PDBs:
```bash
python evaluation.py
```

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


## Results

The table below reproduces Table 2 from the manuscript: reconstruction accuracy on community benchmarks. Metrics are backbone TM-score (↑) and RMSD in Å (↓).

| Dataset    | Metric   | GCP-VQVAE (Ours) | FoldToken-4 | ESM-3 VQVAE | Structure Tokenizer |
|-----------:|:---------|:-----------------|:------------|:------------|:--------------------|
| CASP14     | TM-score | 0.9890           | 0.5410      | 0.5042      | 0.3624              |
|            | RMSD     | 0.5431           | 8.9838      | 10.4611     | 10.5344             |
| CASP15     | TM-score | 0.9884           | 0.3289      | 0.3206      | 0.2329              |
|            | RMSD     | 0.5293           | 14.6702     | 13.1877     | 14.8956             |
| CASP16     | TM-score | 0.9857           | 0.8055      | 0.7685      | 0.6058              |
|            | RMSD     | 0.7567           | 5.5094      | 8.2640      | 8.7106              |
| CAMEO-2024 | TM-score | 0.9918           | 0.4784      | 0.4633      | 0.3575              |
|            | RMSD     | 0.4377           | 12.1089     | 12.1138     | 13.5360             |

Notes:
- FoldToken-4 uses a 256-size vocabulary; others use 4096.
- The Structure Tokenizer baseline supports only sequence lengths 50–512; out-of-range samples are excluded for that column only.
- Evaluation scripts for baselines were reproduced where public tooling was incomplete; see repository docs for details.

## 📜 Citation

If you use this code or the pretrained models, please cite the following paper:

```bibtex
will be added soon
```

