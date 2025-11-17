#!/bin/bash

# Install PyTorch with CUDA 12.9 support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Graph and geometric deep learning libraries
pip install torch_geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu129.html

# Additional PyTorch-related packages
pip install torchmetrics
pip install -U einops
pip install bitsandbytes

# Transformers ecosystem
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

# Visualization and logging
pip install tensorboard

# Data handling & utilities
pip install python-box
pip install h5py
pip install pandas
pip install scikit-learn
pip install joblib

# Specialized libraries
pip install -U vector-quantize-pytorch
pip install -U x_transformers
pip install tmtools
pip install jaxtyping
pip install beartype
pip install omegaconf
pip install ndlinear
pip install torch_tb_profiler
pip install tqdm
pip install biopython
pip install graphein
pip install esm

# Graph analytics libraries from NVIDIA
pip install cugraph-cu12 -f https://pypi.nvidia.com
pip install pylibcugraphops-cu12 -f https://pypi.nvidia.com

# Set environment variable
export TOKENIZERS_PARALLELISM=false

echo "Installation completed successfully!"
