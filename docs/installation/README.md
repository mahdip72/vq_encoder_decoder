# Installation

## Option 1: Using Pre-built Docker Images

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

## Option 2: Building from Dockerfile

```bash
# Clone the repository
git clone https://github.com/mahdip72/vq_encoder_decoder.git
cd vq_encoder_decoder

# Build the Docker image
docker build -t vqvae3d .

# Run the container
docker run --gpus all -it vqvae3d
```

### (Optional, Hopper only) FlashAttention-3
If you are on H100, H800, GH200, H200 (SM90) you can enable FlashAttention-3 for faster, lower-memory attention.

Build with FA3 baked in:
```bash
docker build --build-arg FA3=1 -t vqvae3d-fa3 .
```
