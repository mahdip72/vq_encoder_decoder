#!/bin/bash

# Install PyTorch with CUDA 12.9 support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Graph and geometric deep learning libraries
pip install torch_geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu129.html

# ProteinWorkshop from specific commit
pip install --no-deps git+https://github.com/mahdip72/ProteinWorkshop.git

# Additional PyTorch-related packages
pip install torchmetrics
pip install einops

# Transformers ecosystem
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

# Visualization and logging
pip install tensorboard
pip install plotly

# Data handling & utilities
pip install python-box
pip install h5py
pip install pandas
pip install scikit-learn
pip install joblib

# Specialized libraries
pip install fair-esm
pip install vector-quantize-pytorch
pip install x_transformers
pip install tmtools
pip install jaxtyping
pip install beartype
pip install omegaconf
pip install ndlinear
pip install torch_tb_profiler

# Specific versions from Dockerfile
pip install "graphein==1.7.7"
pip install "loguru==0.7.0"
pip install "fair-esm==2.0.0"
pip install "hydra-core==1.3.2"
pip install "biotite==0.37.0"
pip install "e3nn==0.5.1"
pip install "einops==0.6.1"
pip install "beartype==0.15.0"
pip install "rich==13.5.2"
pip install "pytdc"
pip install "wandb"
pip install "lovely-tensors==0.1.15"
pip install "psutil==5.9.5"
pip install "tqdm==4.66.1"
pip install "jaxtyping==0.2.24"
pip install "omegaconf==2.3.0"
pip install "pytorch-lightning"
pip install "lightning"
pip install "python-dotenv==1.0.0"
pip install "wget==3.2"
pip install "opt-einsum==3.3.0"
pip install "pyrootutils==1.0.4"
pip install "hydra-colorlog==1.2.0"
pip install "bitsandbytes"

# Set environment variable
export TOKENIZERS_PARALLELISM=false

echo "Installation completed successfully!"
