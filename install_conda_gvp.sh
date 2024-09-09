#!/bin/bash

mamba install --yes pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
mamba install --yes cudatoolkit-dev -c senyan.dev
pip install torch_geometric
pip install torchmetrics
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
pip install fair-esm
mamba install --yes pandas
mamba install --yes pytorch-scatter -c pyg
mamba install --yes pytorch-cluster -c pyg
pip install h5py
mamba install --yes matplotlib
mamba install --yes openpyxl
pip install vector-quantize-pytorch
pip install plotly
pip install joblib
pip install tmtools
pip install scikit-learn
pip install schedulefree