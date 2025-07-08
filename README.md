# Vector Quantized Encoder-Decoder for Protein 3D Structures

This research project implements a Vector Quantized Variational Autoencoder (VQ-VAE) for learning discrete representations of protein 3D structures. The model combines geometric deep learning with transformer-based architectures to encode protein structures into discrete tokens and decode them back to 3D coordinates.

## Model Architecture

The system consists of three main components:

1. **Structure Encoder (GCPNet)**: Processes protein backbone atoms (N, CA, C, O) and extracts 128-dimensional structural features
2. **VQ-VAE Transformer**: 
   - Encoder: Projects features through transformer layers to discrete codebook space (4096 codes, 256D)
   - Vector Quantization: Learns discrete representations with commitment and orthogonal regularization
   - Decoder: Reconstructs 3D coordinates from quantized representations
3. **Geometric Decoder**: Outputs backbone coordinates and pairwise relationship predictions

## Installation

### Option 1: Using Pre-built Docker Images

For AMD64 systems:
```bash
docker pull mahdip72/vqvae3d:amd_v3
docker run --gpus all -it mahdip72/vqvae3d:amd_v3
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

### Option 3: Conda Environment Setup

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


## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU
- 16GB+ GPU memory recommended for training

For detailed dependencies, see `Dockerfile` or installation scripts.
