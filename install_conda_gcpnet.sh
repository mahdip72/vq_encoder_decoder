#!/bin/bash

mamba install --yes pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install --yes cudatoolkit-dev -c senyan.dev
mamba install --yes pandas
mamba install --yes pytorch-scatter -c pyg
mamba install --yes pytorch-cluster -c pyg
mamba install --yes matplotlib
mamba install --yes openpyxl
pip install torchtext
pip install torch_geometric
pip install torchmetrics
pip install einops
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
pip install fair-esm
pip install h5py
pip install vector-quantize-pytorch
pip install plotly
pip install joblib
pip install tmtools
pip install scikit-learn
pip install schedulefree
pip install jaxtyping
pip install beartype
pip install omegaconf