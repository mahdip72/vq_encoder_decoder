from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from utils.utils import load_configs
from data.dataset import DistanceMapVQVAEDataset
from utils.metrics import batch_distance_map_to_coordinates


config_path = "configs/config_distance_map_vqvae.yaml"
with open(config_path) as file:
    config_file = yaml.full_load(file)
configs = load_configs(config_file)

configs.train_settings.data_path = "/home/renjz/data/swissprot_1024_h5"

dataset = DistanceMapVQVAEDataset(configs.train_settings.data_path, train_mode=True, rotate_randomly=False,
                                        configs=configs)

dataloader = DataLoader(dataset, batch_size=configs.train_settings.batch_size,
                          shuffle=configs.train_settings.shuffle,
                          num_workers=configs.train_settings.num_workers,
                          pin_memory=False)

print(dataset[0])

for data in dataloader:
    distance_map = data["input_distance_map"]
    print(distance_map.size())
    predicted_coords = batch_distance_map_to_coordinates(distance_map)

    # TODO: Calculate MAE
