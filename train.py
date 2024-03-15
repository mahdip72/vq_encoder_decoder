import argparse
import yaml
import torch
import numpy as np
from utils import load_configs, prepare_saving_dir


def train_loop():
    pass


def evaluate_loop():
    pass


def main(dict_config, config_file_path):
    configs = load_configs(dict_config)

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    for epoch in range(1, configs.train_settings.num_epochs + 1):
        train_loop()
        evaluate_loop()

    print("Train finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deep neural nets.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
