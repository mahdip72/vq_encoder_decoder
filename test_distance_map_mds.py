from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm
import yaml
from utils.utils import load_configs
from data.dataset import DistanceMapVQVAEDataset
from utils.metrics import batch_distance_map_to_coordinates


config_path = "configs/config_distance_map_vqvae.yaml"
with open(config_path) as file:
    config_file = yaml.full_load(file)
configs = load_configs(config_file)

configs.train_settings.data_path = "/home/renjz/data/cath_4_3_0"

dataset = DistanceMapVQVAEDataset(configs.train_settings.data_path, train_mode=True, rotate_randomly=False,
                                        configs=configs)

dataloader = DataLoader(dataset, batch_size=configs.train_settings.batch_size,
                          shuffle=configs.train_settings.shuffle,
                          num_workers=configs.train_settings.num_workers,
                          pin_memory=False)

mae = MeanAbsoluteError()

# Get batches of distance maps from h5 files
progress_bar = tqdm(dataloader)
for data in progress_bar:

    distance_map = data["input_distance_map"][:,0,:,:]
    target_coords = data["target_coords"]

    # Reconstruct coordinates from distance map using MDS
    rec_coords = batch_distance_map_to_coordinates(distance_map)

    # Calculate mean absolute error between reconstructed and target coordinates
    mae.update(rec_coords, target_coords)
    current_mae = mae.compute().item()

    progress_bar.set_postfix({"mae": current_mae})

overall_mae = mae.compute().item()
print("Overall MAE:", overall_mae)
