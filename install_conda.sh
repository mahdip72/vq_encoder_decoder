#!/bin/bash

conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torchmetrics
conda install --yes matplotlib
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
conda install --yes pandas
pip install -U scikit-learn
pip install rdkit-pypi
pip install schedulefree
pip install vector-quantize-pytorch
pip install plotly
pip install joblib
pip install egnn_pytorch

# for multi-gpu training
export TOKENIZERS_PARALLELISM=false