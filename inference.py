import numpy as np
import torch
import cv2
from utils.utils import load_checkpoints_simple
from data.data_cifar import prepare_dataloaders
from models.model import prepare_models


def main():
    import yaml
    from utils.utils import load_configs

    config_path = "results/2024-04-22__21-53-27/config_gvp.yaml"
    checkpoint_path = "results/2024-04-22__21-53-27/checkpoints/epoch_24.pth"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    configs = load_configs(config_file)


if __name__ == '__main__':
    main()
