#!/bin/bash

# 1. Remove the FA2 wheel that was pip-installed in the Dockerfile
pip uninstall -y flash-attn

# 2. Clone (or fetch) the repo
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper

# 3. Make sure build deps are present
pip install --upgrade ninja packaging einops

# 5. Ask PyTorch to target Hopper
export TORCH_CUDA_ARCH_LIST="90"

# 6. Build & install FA3
python setup.py install            # or: pip install . --no-build-isolation