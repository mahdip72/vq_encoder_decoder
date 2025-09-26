FROM nvcr.io/nvidia/pytorch:25.06-py3

# Use bash as default shell
SHELL ["/bin/bash", "-c"]

# (Optional) system utilities similar to previous image layer; can be removed if undesired
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tmux \
        htop \
        nvtop \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# (Optional) upgrade pip first (not in install.sh but kept for robustness)
RUN pip3 install --no-cache-dir --upgrade pip

# -----------------------------
# Install PyTorch (matches install.sh order)
# -----------------------------
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu129
RUN pip3 install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cu129

# -----------------------------
# Graph and geometric deep learning libraries
# -----------------------------
RUN pip3 install --no-cache-dir torch_geometric
RUN pip3 install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
RUN pip3 install --no-cache-dir torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu129.html

# -----------------------------
# Additional PyTorch-related packages
# -----------------------------
RUN pip3 install --no-cache-dir torchmetrics
RUN pip3 install --no-cache-dir -U einops
RUN pip3 install --no-cache-dir bitsandbytes

# -----------------------------
# Transformers ecosystem
# -----------------------------
RUN pip3 install --no-cache-dir accelerate
RUN pip3 install --no-cache-dir transformers
RUN pip3 install --no-cache-dir timm
RUN pip3 install --no-cache-dir 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

# -----------------------------
# Visualization and logging
# -----------------------------
RUN pip3 install --no-cache-dir tensorboard

# -----------------------------
# Data handling & utilities
# -----------------------------
RUN pip3 install --no-cache-dir python-box
RUN pip3 install --no-cache-dir h5py
RUN pip3 install --no-cache-dir pandas
RUN pip3 install --no-cache-dir scikit-learn
RUN pip3 install --no-cache-dir joblib

# -----------------------------
# Specialized libraries
# -----------------------------
RUN pip3 install --no-cache-dir -U vector-quantize-pytorch
RUN pip3 install --no-cache-dir -U x_transformers
RUN pip3 install --no-cache-dir tmtools
RUN pip3 install --no-cache-dir jaxtyping
RUN pip3 install --no-cache-dir beartype
RUN pip3 install --no-cache-dir omegaconf
RUN pip3 install --no-cache-dir ndlinear
RUN pip3 install --no-cache-dir torch_tb_profiler
RUN pip3 install --no-cache-dir tqdm
RUN pip3 install --no-cache-dir biopython
RUN pip3 install --no-cache-dir graphein

# Environment variables
ENV TOKENIZERS_PARALLELISM=false
