# GCP-VQVAE: A Geometry-Complete Language for Protein 3D Structure

## Abstract

Converting protein tertiary structure into discrete tokens via vector-quantized variational autoencoders (VQVAEs) creates a language of 3D geometry and provides a natural interface between sequence and structure models. While pose invariance is commonly enforced, retaining chirality and directional cues without sacrificing reconstruction accuracy remains challenging. In this paper, we introduce GCP-VQVAE, a geometry-complete tokenizer built around a strictly SE(3)-equivariant GCPNet encoder that preserves orientation and chirality of protein backbone. We vector-quantize pose-invariant readouts into a 4096-token vocabulary, and a transformer decoder maps tokens back to backbone coordinates via a 6D rotation head trained with SE(3)-invariant objectives. 

Building on these properties, we train GCP‑VQVAE on a corpus of 24 million monomer protein backbone structures gathered from the AlphaFold Protein Structure Database. On the CAMEO‑2024, CASP15, and CASP16 evaluation datasets, the model achieves backbone RMSDs of 0.4377\,Å, 0.5293\,Å, and 0.7576\,Å, respectively, and achieves 100\% codebook utilization on a held‑out validation set, substantially outperforming prior VQ‑VAE–based tokenizers and achieving state-of-the-art performance. Lastly, we elaborate on the various applications of this foundation-like model, such as protein structure compression and the integration of generative AI models.


## News
- 




## Model Architecture

The system consists of three main components:

1. **Structure Encoder (GCPNet)**: Processes protein backbone atoms (N, CA, C, O) and extracts 128-dimensional structural features
2. **VQ-VAE Transformer**: 
   - Encoder: Projects features through transformer layers to discrete codebook space (4096 codes, 256D)
   - Vector Quantization: Learns discrete representations with commitment and orthogonal regularization
   - Decoder: Reconstructs 3D coordinates from quantized representations
3. **Geometric Decoder**: Outputs backbone coordinates and pairwise relationship predictions

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

For encoding proteins to discrete tokens:
```bash
python inference_encode.py --config_path configs/inference_encode_config.yaml
```

For decoding tokens back to 3D structures:
```bash
python inference_decode.py --config_path configs/inference_decode_config.yaml
```



