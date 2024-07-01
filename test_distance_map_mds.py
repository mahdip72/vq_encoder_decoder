from pathlib import Path
from scipy.spatial import procrustes
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm
import yaml

from utils.utils import load_configs
from data.dataset import DistanceMapVQVAEDataset
from data.normalizer import Protein3DProcessing
from utils.metrics import batch_distance_map_to_coordinates


def calc_dist(pt1, pt2):
    """
    Calculate the distance between two 3D points
    :param pt1: (torch.tensor) coordinates of first point
    :param pt2: (torch.tensor) coordinates of second point
    :return: (float) distance between the two points
    """
    x = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2
    return torch.sqrt(x).item()


def transform_rec_coords(rec_coords, target_coords):
    """
    Transform a batch of reconstructed coordinates by translating the coordinates toward the target coordinates
    :param rec_coords: (torch.tensor) batch of reconstructed coordinates
    :param target_coords: (torch.tensor) batch of target coordinates
    :return new_coords: (torch.tensor) transformed coordinates
    """

    new_coords = torch.clone(rec_coords)

    for i in range(len(new_coords)):

        # Distance between the first reconstructed point and first target point of the current sample
        dist = calc_dist(new_coords[i][0], target_coords[i][0])
        dist_tensor = torch.tensor([dist, dist, dist])

        for j in range(len(new_coords[i])):
            new_coords[i][j] -= dist_tensor

    return new_coords


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

# Prepare the normalizer for denormalization
processor = Protein3DProcessing()
processor.load_normalizer(configs.normalizer_path)

# Prepare processor for normalizing coordinates
# coords_processor = Protein3DProcessing()
# coords_processor.load_normalizer("data/normalizer.pkl")

mae = MeanAbsoluteError()

# Get batches of distance maps from h5 files
progress_bar = tqdm(dataloader)
for data in progress_bar:

    distance_maps = data["input_distance_map"][:,0,:,:]
    target_coords = data["target_coords"]

    mds_args = {'n_components': 3, 'dissimilarity': 'precomputed', 'random_state': 42,
                'n_init': 10, 'max_iter': 300, 'eps': 1e-9, 'n_jobs': -1}

    # Denormalize distance maps
    for i in range(len(distance_maps)):
        distance_maps[i] = processor.denormalize_distance_map(distance_maps[i])

    # Reconstruct coordinates from distance map using MDS
    rec_coords = batch_distance_map_to_coordinates(distance_maps, n_components=mds_args['n_components'],
                                                   dissimilarity=mds_args['dissimilarity'],
                                                   random_state=mds_args['random_state'],
                                                   n_init=mds_args['n_init'], max_iter=mds_args['max_iter'],
                                                   eps=mds_args['eps'], n_jobs=mds_args['n_jobs'])

    # Perform Procrustes analysis on coordinates
    for i in range(len(target_coords)):
        new_target, new_rec, disparity = procrustes(target_coords[i], rec_coords[i])
        target_coords[i] = torch.tensor(new_target)
        rec_coords[i] = torch.tensor(new_rec)

    # Calculate mean absolute error between reconstructed and target coordinates
    mae.update(rec_coords, target_coords)
    current_mae = mae.compute().item()

    progress_bar.set_postfix({"mae": current_mae})

    """
    print("After Procrustes")
    print(target_coords[0][0:3])
    print(rec_coords[0][0:3])
    #print(calc_dist(target_coords[0][0], target_coords[0][1]))
    #print(calc_dist(rec_coords[0][0], rec_coords[0][1]))

    print('\n')
    #print(distance_maps[0][0][0:3])
    exit()
    """

overall_mae = mae.compute().item()
print("Overall MAE:", overall_mae)
