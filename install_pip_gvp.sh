#!/bin/bash

pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics
uv pip install h5py
uv pip install matplotlib
uv pip install pandas
uv pip install accelerate
uv pip install transformers
uv pip install timm
uv pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
uv pip install tensorboard
uv pip install python-box
uv pip install peft
uv pip install -U scikit-learn
uv pip install rdkit-pypi
uv pip install vector-quantize-pytorch
