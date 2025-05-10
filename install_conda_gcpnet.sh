#!/bin/bash

mamba install --yes pandas
mamba install --yes matplotlib
mamba install --yes openpyxl
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install git+https://github.com/a-r-j/ProteinWorkshop.git@da7cfe6d3e469ef64d4899dc31a9391a3b69c8cc
pip install torchtext
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
pip install ndlinear
pip install torch_tb_profiler