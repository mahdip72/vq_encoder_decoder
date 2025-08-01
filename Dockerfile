FROM nvcr.io/nvidia/pytorch:24.10-py3

# Use bash as default shell
SHELL ["/bin/bash", "-c"]


# Graph and geometric deep learning libraries
RUN pip install torch_geometric
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu126.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu126.html

# ProteinWorkshop from specific commit
RUN pip install --no-deps git+https://github.com/a-r-j/ProteinWorkshop.git@da7cfe6d3e469ef64d4899dc31a9391a3b69c8cc

# Additional PyTorch-related packages
# RUN pip install torchtext
RUN pip install torchmetrics
RUN pip install einops

# Transformers ecosystem
RUN pip install accelerate
RUN pip install transformers
RUN pip install timm
RUN pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

# Visualization and logging
RUN pip install tensorboard
RUN pip install plotly

# Data handling & utilities
RUN pip install python-box
RUN pip install h5py
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install joblib

# Specialized libraries
RUN pip install fair-esm
RUN pip install vector-quantize-pytorch
RUN pip install x_transformers
RUN pip install tmtools
RUN pip install schedulefree
RUN pip install jaxtyping
RUN pip install beartype
RUN pip install omegaconf
RUN pip install ndlinear
RUN pip install torch_tb_profiler

RUN python3 -m pip install "graphein==1.7.5"
RUN python3 -m pip install "loguru==0.7.0"
RUN python3 -m pip install "fair-esm==2.0.0"
RUN python3 -m pip install "hydra-core==1.3.2"
RUN python3 -m pip install "biotite==0.37.0"
RUN python3 -m pip install "e3nn==0.5.1"
RUN python3 -m pip install "einops==0.6.1"
RUN python3 -m pip install "beartype==0.15.0"
RUN python3 -m pip install "rich==13.5.2"
RUN python3 -m pip install "pytdc==0.4.1"
RUN python3 -m pip install "wandb==0.15.8"
RUN python3 -m pip install "lovely-tensors==0.1.15"
RUN python3 -m pip install "psutil==5.9.5"
RUN python3 -m pip install "tqdm==4.66.1"
RUN python3 -m pip install "jaxtyping==0.2.24"
RUN python3 -m pip install "omegaconf==2.3.0"
RUN python3 -m pip install "pytorch-lightning==2.0.7"
RUN python3 -m pip install "lightning==2.0.7"
RUN python3 -m pip install "python-dotenv==1.0.0"
RUN python3 -m pip install "wget==3.2"
RUN python3 -m pip install "opt-einsum==3.3.0"
RUN python3 -m pip install "pyrootutils==1.0.4"
RUN python3 -m pip install "hydra-colorlog==1.2.0"
RUN python3 -m pip install "bitsandbytes"


# Environment variables
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tmux \
        htop \
        nvtop \
 && apt-get clean && rm -rf /var/lib/apt/lists/*