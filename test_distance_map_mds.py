from pathlib import Path
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanAbsoluteError
from tqdm import tqdm
import yaml
import os
import glob

from utils.utils import load_configs
from utils.utils import load_h5_file

from data.data_contactmap import DistanceMapVQVAEDataset
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


class SimpleDistanceMapDataset(Dataset):
    """
    Simplified dataset for converting h5 files to distance map (no normalization)
    """
    def __init__(self, data_path, configs):
        """
        :param data_path: (string or Path) path to directory of h5 files
        :param configs: (Box) configurations
        """
        self.max_task_samples = configs.train_settings.max_task_samples
        self.max_length = configs.model.max_length
        self.data_path = str(data_path)
        self.h5_samples = glob.glob(os.path.join(self.data_path, '*.h5'))[:self.max_task_samples]

    def __len__(self):
        return len(self.h5_samples)

    @staticmethod
    def handle_nan_coordinates(coords: torch.Tensor) -> torch.Tensor:
        """
        Replaces NaN values in the coordinates with the previous or next valid coordinate values.

        Parameters:
        -----------
        coords : torch.Tensor
            A tensor of shape (N, 4, 3) representing the coordinates of a protein structure.

        Returns:
        --------
        torch.Tensor
            The coordinates with NaN values replaced by the previous valid coordinate values.
        """
        # Flatten the coordinates for easier manipulation
        original_shape = coords.shape
        coords = coords.view(-1, 3)

        # check if there are any NaN values in the coordinates
        while torch.isnan(coords).any():
            # Identify NaN values
            nan_mask = torch.isnan(coords)

            if not nan_mask.any():
                return coords.view(original_shape)  # Return if there are no NaN values

            # Iterate through coordinates and replace NaNs with the previous valid coordinate
            for i in range(1, coords.shape[0]):
                if nan_mask[i].any() and not torch.isnan(coords[i - 1]).any():
                    coords[i] = coords[i - 1]

            for i in range(0, coords.shape[0] - 1):
                if nan_mask[i].any() and not torch.isnan(coords[i + 1]).any():
                    coords[i] = coords[i + 1]

        return coords.view(original_shape)

    @staticmethod
    def create_distance_map(coords):
        """
        Computes the pairwise distance map for CA coordinates using PyTorch.

        Parameters:
        ca_coords (torch.Tensor): A 2D tensor of shape (n, 3) containing the coordinates of CA atoms.

        Returns:
        torch.Tensor: A 2D tensor of shape (n, n) containing the pairwise distances.
        """
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        distance_map = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        return distance_map

    def __getitem__(self, idx):

        sample_path = self.h5_samples[idx]
        sample = load_h5_file(sample_path)
        basename = os.path.basename(sample_path)
        pid = basename.split('.h5')[0].split('_')[0]
        coords_list = sample[1].tolist()
        coords_tensor = torch.Tensor(coords_list)

        coords_tensor = coords_tensor[:self.max_length, ...]

        coords_tensor = self.handle_nan_coordinates(coords_tensor)
        coords_tensor = coords_tensor[..., 3:6].reshape(1, -1, 3)
        distance_map = self.create_distance_map(coords_tensor.squeeze(0))

        # expand the first dimension of distance maps
        distance_map = distance_map.unsqueeze(0)

        return {'pid': pid, 'coords': coords_tensor.squeeze(0), 'distance_map': distance_map}


def prepare_dataloaders(configs):
    """
    Prepare distance map dataloaders.
    :param configs: configurations for contact map
    :return: train dataloader
    """
    data_path = configs.train_settings.data_path

    dataset = DistanceMapVQVAEDataset(data_path, train_mode=True, rotate_randomly=False, configs=configs)
    # dataset = SimpleDistanceMapDataset(data_path, configs)

    dataloader = DataLoader(dataset, batch_size=configs.train_settings.batch_size,
                              shuffle=configs.train_settings.shuffle,
                              num_workers=configs.train_settings.num_workers,
                              pin_memory=False)

    return dataloader


def custom_procrustes(data1, data2):
    """
    Perform Procrustes analysis on two datasets. Regular Procrustes analysis returns
    a centered and standardized version of the first matrix, but this function
    recovers the original first matrix with the second matrix adjusted accordingly.
    :param data1: (array_like) first matrix
    :param data2: (array_like) second matrix
    :return:
        mtx1 (array_like) original first matrix
        mtx2 (array_like) the orientation of `data2` that best fits `data1`
        disparity (float) total squared error between the final matrices
    """

    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mean1 = np.mean(mtx1, 0)
    mean2 = np.mean(mtx2, 0)
    mtx1 -= mean1
    mtx2 -= mean2

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    # Rescale both matrices to their original size
    mtx1 *= norm1
    mtx2 *= norm2

    # Translate both matrices to match the original first matrix
    mtx1 += mean1
    mtx2 += mean1

    return mtx1, mtx2, disparity


if __name__ == "__main__":

    config_path = "configs/config_distance_map_vqvae.yaml"
    with open(config_path) as file:
        config_file = yaml.full_load(file)
    main_configs = load_configs(config_file)

    main_configs.train_settings.data_path = "/home/renjz/data/cath_4_3_0"

    # # Prepare the normalizer for denormalization
    # processor = Protein3DProcessing()
    # processor.load_normalizer(main_configs.normalizer_path)
    #
    # # Prepare processor for normalizing coordinates
    # # coords_processor = Protein3DProcessing()
    # # coords_processor.load_normalizer("data/normalizer.pkl")

    dmap_dataloader = prepare_dataloaders(main_configs)

    mae = MeanAbsoluteError()

    # Get batches of distance maps from h5 files
    progress_bar = tqdm(dmap_dataloader)
    for data in progress_bar:

        distance_maps = data["input_distance_map"][:, 0, :, :]
        target_coords = data["target_coords"]

        mds_args = {'n_components': 3, 'dissimilarity': 'precomputed', 'random_state': 42,
                    'n_init': 4, 'max_iter': 300, 'eps': 1e-7, 'n_jobs': -1}

        # # Denormalize distance maps
        # for i in range(len(distance_maps)):
        #     distance_maps[i] = processor.denormalize_distance_map(distance_maps[i])

        # Reconstruct coordinates from distance map using MDS
        rec_coords = batch_distance_map_to_coordinates(distance_maps, n_components=mds_args['n_components'],
                                                       dissimilarity=mds_args['dissimilarity'],
                                                       random_state=mds_args['random_state'],
                                                       n_init=mds_args['n_init'], max_iter=mds_args['max_iter'],
                                                       eps=mds_args['eps'], n_jobs=mds_args['n_jobs'])

        # Perform Procrustes analysis on coordinates
        for i in range(len(target_coords)):
            # new_target, new_rec, disparity = procrustes(target_coords[i], rec_coords[i])
            new_target, new_rec, disparity = custom_procrustes(target_coords[i], rec_coords[i])
            # target_coords[i] = torch.tensor(new_target)
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
