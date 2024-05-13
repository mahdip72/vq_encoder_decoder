#!/bin/bash

conda install --yes pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install --yes pyg -c pyg
pip install torchmetrics
pip install accelerate
pip install transformers
pip install timm
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install tensorboard
pip install python-box
pip install fair-esm
conda install --yes pandas
pip install peft
conda install --yes pytorch-scatter -c pyg
conda install --yes pytorch-cluster -c pyg
pip install h5py
conda install --yes matplotlib
conda install --yes openpyxl