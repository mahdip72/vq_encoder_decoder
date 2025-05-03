#!/bin/bash

mamba install --yes pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
mamba install --yes cudatoolkit-dev -c senyan.dev
mamba install --yes matplotlib
mamba install --yes pandas
pip install torchmetrics
pip install torchtext
pip install torch_geometric
pip install einops
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
pip install rdkit-pypi
pip install schedulefree
pip install vector-quantize-pytorch
pip install plotly
pip install joblib
pip install egnn_pytorch
pip install tmtools
pip install flash_attn
pip install scikit-learn
pip install schedulefree
pip install h5py
pip install -U scikit-learn
pip install ndlinear

# for multi-gpu training
export TOKENIZERS_PARALLELISM=false