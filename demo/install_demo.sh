#!/usr/bin/env bash
set -euo pipefail

# Minimal dependency install for the demo evaluation pipeline.
# Adjust CUDA wheel URLs if you are on a different CUDA version or CPU only.

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# PyG stack (torch_geometric + required ops)
pip install torch_geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu129.html

# Core demo dependencies
pip install numpy scipy pyyaml python-box h5py tqdm biopython
pip install graphein
pip install transformers
pip install vector-quantize-pytorch x_transformers ndlinear
pip install jaxtyping typing_extensions

echo "Demo dependencies installed."
