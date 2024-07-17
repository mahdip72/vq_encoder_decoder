import numpy as np
from tmtools import tm_align
import torch.nn as nn
from torchmetrics import Metric
import torch
from sklearn.manifold import MDS
from utils.utils import load_h5_file


def ensure_symmetry_torch_upper_to_lower(tensor):
    """
    Ensures that the input tensor is symmetric by copying the upper triangle to the lower triangle.

    Args:
        tensor (torch.Tensor): Input distance matrix of shape (m x m).

    Returns:
        torch.Tensor: Symmetric distance matrix.
    """
    # Get the upper triangle indices, excluding the diagonal
    triu_indices = torch.triu_indices(tensor.shape[-2], tensor.shape[-1], offset=1)

    # Copy the upper triangle to the lower triangle
    tensor[..., triu_indices[1], triu_indices[0]] = tensor[..., triu_indices[0], triu_indices[1]]

    return tensor


def batch_distance_map_to_coordinates(batch_distance_map, **kwargs):
    """
    Converts a batch of distance maps from a PyTorch tensor format to 3D coordinates.

    Args:
        batch_distance_map (torch.Tensor): A (b x m x m) batch of distance matrices in PyTorch tensor format.
        **kwargs: non-default arguments for MDS (optional)

    Returns:
        torch.Tensor: A (b x m x 3) tensor of 3D coordinates.
    """
    # Get the batch size
    batch_size = batch_distance_map.size(0)
    num_points = batch_distance_map.size(1)

    # Initialize an empty list to hold the coordinates
    all_coordinates = []

    # Default MDS arguments if kwargs were not provided
    if len(kwargs) == 0:
        mds_args = {'n_components': 3, 'dissimilarity': 'precomputed', 'random_state': 42,
                  'n_init': 2, 'max_iter': 96, 'eps': 1e-3, 'n_jobs': -1}

    else:
        mds_args = kwargs

    # Loop over each distance map in the batch
    for i in range(batch_size):
        distance_matrix_np = ensure_symmetry_torch_upper_to_lower(batch_distance_map[i].cpu().numpy())

        mds = MDS(n_components=mds_args['n_components'], dissimilarity=mds_args['dissimilarity'], random_state=mds_args['random_state'],
                  n_init=mds_args['n_init'], max_iter=mds_args['max_iter'], eps=mds_args['eps'], n_jobs=mds_args['n_jobs'])

        # Fit the model to the distance matrix
        coordinates_np = mds.fit_transform(distance_matrix_np)

        # Convert the coordinates to a PyTorch tensor and append to the list
        coordinates_tensor = torch.tensor(coordinates_np, dtype=torch.float32)
        all_coordinates.append(coordinates_tensor)

    # Stack all the coordinates to form a batch
    batch_coordinates = torch.stack(all_coordinates)

    return batch_coordinates


def perform_tm_align(coords1, coords2, seq1, seq2):
    """
    Perform TM alignment on two protein structures and calculate the TM-score.
    The score is normalized based on the first structure. 
    :param coords1: (torch.Tensor) list of 3D coordinates of the first structure
    :param coords2: (torch.Tensor) list of 3D coordinates of the second structure
    :param seq1: (str) sequence of first structure
    :param seq2: (str) sequence of second structure
    :return: (float) TM-score normalized based on first structure
    """
    coords1 = coords1.numpy()
    coords2 = coords2.numpy()

    # Align the two structures
    tm_result = tm_align(coords1, coords2, seq1, seq2)

    # Return the TM score, normalized by the length of the first sequence
    return tm_result.tm_norm_chain1


def tm_from_h5(h5_file1, h5_file2):
    """
    Calculate a TM-score from the h5 files of two protein structures.
    :param h5_file1: (Path or str) path to h5 file for first protein structure
    :param h5_file2: (Path or str) path to h5 file for second protein structure
    :return tm_score: (float) TM-score for the protein structures
    """
    seq1, n_ca_c_o_coord1, plddt_scores = load_h5_file(h5_file1)
    seq2, n_ca_c_o_coord2, plddt_scores = load_h5_file(h5_file2)

    # Get the alpha carbon coordinates
    ca_coords1 = torch.from_numpy(n_ca_c_o_coord1[:,1,:])
    ca_coords2 = torch.from_numpy(n_ca_c_o_coord2[:,1,:])

    # Convert the sequences from byte strings to strings
    seq1 = str(seq1, encoding="utf-8")
    seq2 = str(seq2, encoding="utf-8")

    # Calculate TM-score
    tm_score = perform_tm_align(ca_coords1, ca_coords2, seq1, seq2)
    return tm_score


def batch_tm_score(coords_batch1, coords_batch2, seqs1, seqs2, masks1=None, masks2=None):
    """
    Calculate the average TM-score between two batches of coordinates.
    :param coords_batch1: (torch.Tensor) first batch of 3D coordinates
    :param coords_batch2: (torch.Tensor) second batch of 3D coordinates
    :param seqs1: (list[str]) list of sequences for first batch
    :param seqs2: (list[str]) list of sequences for second batch
    :param masks1: (torch.Tensor) masks for first batch, optional
    :param masks2: (torch.Tensor) masks for second batch, optional
    :return: (float) average TM-score
    """

    # Divide masks and batches into individual structures
    # Apply mask to each structure
    # Shorten sequence to match length of structure, if necessary
    # Calculate TM

    assert coords_batch1.size() == coords_batch2.size()

    total_tm_score = 0.0
    num_samples = 0

    # Iterate through each sample in batch
    for i in range(len(coords_batch1)):
        coords1 = coords_batch1[i]
        coords2 = coords_batch2[i]
        seq1 = seqs1[i]
        seq2 = seqs2[i]

        # Remove padding
        if (masks1 is not None) and (masks2 is not None):
            coords1 = coords1[masks1[i]]
            coords2 = coords2[masks2[i]]

        len_coords1 = len(coords1)
        len_coords2 = len(coords2)

        # Remove any excess residues from the sequences
        seq1 = seq1[0:len_coords1]
        seq2 = seq2[0:len_coords2]
        assert len(seq1) == len_coords1
        assert len(seq2) == len_coords2

        # Calculate TM-score
        tm_score = perform_tm_align(coords1, coords2, seq1, seq2)
        total_tm_score += tm_score
        num_samples += 1

    avg_tm_score = total_tm_score / num_samples
    return avg_tm_score


class GDTTS(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_gdt", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape, "Predictions and target must have the same shape"

        thresholds = [1.0, 2.0, 4.0, 8.0]
        gdt_scores = torch.zeros(len(thresholds))

        distances = torch.sqrt(torch.sum((preds - target) ** 2, dim=-1))

        for i, threshold in enumerate(thresholds):
            gdt_scores[i] = (distances < threshold).float().mean()

        self.sum_gdt += gdt_scores.mean()
        self.total += 1

    def compute(self):
        return self.sum_gdt / self.total


class LDDT(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_lDDT", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Compute the lDDT score for the predicted and true structures
        lddt_score = self.compute_lddt(preds, target)
        self.sum_lDDT += lddt_score
        self.total += 1

    def compute(self):
        # Compute the final lDDT score based on the state variables
        return self.sum_lDDT / self.total

    @staticmethod
    def compute_lddt(preds: torch.Tensor, target: torch.Tensor, thresholds=(0.5, 1, 2, 4)):
        """
        Compute the local Distance Difference Test (lDDT) score for the predicted and true protein structures.

        Parameters:
        preds (torch.Tensor): The predicted protein structures. Shape: (N, 3) where N is the number of atoms.
        target (torch.Tensor): The true protein structures. Shape: (N, 3) where N is the number of atoms.
        thresholds (tuple, optional): The distance thresholds for the lDDT score. Default is (0.5, 1, 2, 4).

        Returns:
        lddt_score (torch.Tensor): The computed lDDT score. A single floating point number.
        """
        # Ensure the tensors are on the same device and have the same shape
        assert preds.device == target.device
        assert preds.shape == target.shape

        # Compute the pairwise distance matrices for the predicted and actual structures
        preds_dist = torch.cdist(preds, preds)
        target_dist = torch.cdist(target, target)

        # Initialize a tensor to hold the per-atom lDDT scores
        lddt_scores = torch.zeros(preds.shape[0], device=preds.device)

        # For each atom, compute the lDDT score
        for i in range(preds.shape[0]):
            # Get the neighbors within a 15 Ångström radius
            preds_neighbors = preds_dist[i] <= 15
            target_neighbors = target_dist[i] <= 15

            # Compute the lDDT score for each threshold
            for threshold in thresholds:
                # Compute the fraction of distances in the predicted structure that fall within the threshold
                # of the distances in the actual structure
                lddt_scores[i] += ((preds_dist[i, preds_neighbors] - target_dist[i, target_neighbors]).abs() < threshold).float().mean()

            # Average the lDDT scores across all thresholds
            lddt_scores[i] /= len(thresholds)

        # Average the per-atom lDDT scores across all atoms to get the overall lDDT score
        lddt_score = lddt_scores.mean()

        return lddt_score


if __name__ == "__main__":
    # Example usage of MDS
    batch_distance_map_example = torch.tensor([
        [
            [0, 3, 4, 5],
            [3, 0, 2, 4],
            [4, 2, 0, 3],
            [5, 4, 3, 0]
        ],
        [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0]
        ]
    ], dtype=torch.float32)

    batch_coordinates = batch_distance_map_to_coordinates(batch_distance_map_example)
    print("Batch Coordinates:\n", batch_coordinates)

    # Example usage
    mean_lddt = LDDT()

    # Example true and predicted values (coordinates)
    y_true = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred = torch.tensor([[1.1, 0.2, 9.1], [3.9, 4.9, 5.9]])

    # Update the metric with predictions and true values
    mean_lddt.update(y_pred, y_true)

    # Compute the final GDT-TS score
    mean_lddt_score = mean_lddt.compute()
    print(mean_lddt_score.item())
