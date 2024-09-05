#!/bin/bash

mamba install --yes pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
mamba install --yes cudatoolkit-dev -c senyan.dev
pip install torchmetrics
mamba install --yes matplotlib
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
mamba install --yes pandas
pip install -U scikit-learn
pip install rdkit-pypi
pip install schedulefree
pip install vector-quantize-pytorch
pip install plotly
pip install joblib
pip install egnn_pytorch
pip install tmtools
pip3 install flash_attn

# for multi-gpu training
export TOKENIZERS_PARALLELISM=false