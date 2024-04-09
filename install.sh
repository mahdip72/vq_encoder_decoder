#!/bin/bash

conda install --yes pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torchmetrics
conda install --yes h5py
conda install --yes matplotlib
conda install --yes pandas
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
pip install peft
pip install -U scikit-learn
pip install rdkit-pypi
