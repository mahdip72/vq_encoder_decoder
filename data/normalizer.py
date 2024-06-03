import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from utils.utils import load_h5_file
from torch.utils.data import DataLoader, Dataset
import os
import glob
import random


class NormalizerDataset(Dataset):
    """
    This class is a subclass of `torch.utils.data.Dataset` and is used to transform JSON/dictionary-style
    protein structures into featurized protein graphs. The transformation process is described in detail in the
    associated manuscript.

    The transformed protein graphs are instances of `torch_geometric.data.Data` and have the following attributes:
    - x: Alpha carbon coordinates. This is a tensor of shape [n_nodes, 3].
    - seq: Protein sequence converted to an integer tensor according to `self.letter_to_num`. This is a tensor of shape [n_nodes].
    - name: Name of the protein structure. This is a string.
    - node_s: Node scalar features. This is a tensor of shape [n_nodes, 6].
    - node_v: Node vector features. This is a tensor of shape [n_nodes, 3, 3].
    - edge_s: Edge scalar features. This is a tensor of shape [n_edges, 32].
    - edge_v: Edge scalar features. This is a tensor of shape [n_edges, 1, 3].
    - edge_index: Edge indices. This is a tensor of shape [2, n_edges].
    - mask: Node mask. This is a boolean tensor where `False` indicates nodes with missing data that are excluded from message passing.

    This class uses portions of code from https://github.com/jingraham/neurips19-graph-protein-design.

    Parameters:
    - data_list: directory of h5 files.
    - num_positional_embeddings: The number of positional embeddings to use.
    - top_k: The number of edges to draw per node (as destination node).
    - device: The device to use for preprocessing. If "cuda", preprocessing will be done on the GPU.
    """

    def __init__(self, data_path, **kwargs):
        super(NormalizerDataset, self).__init__()

        self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))
        random.shuffle(self.h5_samples)
        # self.h5_samples = self.h5_samples[:300000]

    def __len__(self):
        return len(self.h5_samples)

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        sample = load_h5_file(sample_path)
        coords_list = sample[1].tolist()
        coords_tensor = torch.Tensor(coords_list)
        return coords_tensor


class Protein3DProcessing:
    """
    A class to process protein 3D structures using PCA for rotation standardization and normalization.

    This class provides methods to fit normalization models, apply PCA transformations individually
    for each protein to achieve rotation invariance, and reverse the transformations (denormalize).

    Attributes:
    -----------
    normalizer : StandardScaler
        The normalizer used for coordinate normalization.

    Methods:
    --------
    fit_normalizer(coords_list)
        Fits the normalizer on the given list of coordinates after individual PCA transformation.
    apply_pca(coords)
        Applies PCA transformation individually to standardize the rotation of the coordinates.
    normalize_coords(coords)
        Normalizes the individually PCA-transformed coordinates using the fitted normalizer.
    denormalize_coords(coords)
        Denormalizes the coordinates using the fitted normalizer and reverses individual PCA transformation.
    save_normalizer(file_path)
        Saves the fitted normalizer to a file.
    load_normalizer(file_path)
        Loads the fitted normalizer from a file.

    Usage Example:
    --------------
    import torch
    coords_list = [
        torch.rand((10, 4, 3)),  # Example tensor for one protein structure
        torch.rand((15, 4, 3)),  # Example tensor for another protein structure
        # Add more tensors as needed
    ]
    processor = Protein3DProcessing()
    processor.fit_normalizer(coords_list)
    processor.save_normalizer("normalizer.pkl")
    processor.load_normalizer("normalizer.pkl")
    new_coords = torch.rand((12, 4, 3))  # Example new protein structure
    normalized_coords = processor.normalize_coords(new_coords)
    denormalized_coords = processor.denormalize_coords(normalized_coords)
    """

    def __init__(self, normalizer=None):
        self.normalizer = normalizer

    def fit_normalizer(self, coords_list: list, batch_size: int = 32000):
        """
        Fit a normalizer on the given list of individually PCA-transformed coordinates using batches.

        Parameters:
        -----------
        coords_list : list
            A list of torch.Tensors, each of shape (N, 4, 3) representing the coordinates
            of multiple protein structures.
        batch_size : int, optional
            The size of each batch for fitting the normalizer. Default is 32000.
        """
        # Initialize the normalizer
        self.normalizer = StandardScaler()

        # Apply individual PCA to each set of coordinates and fit the normalizer in batches
        for i in range(0, len(coords_list), batch_size):
            batch_coords_list = coords_list[i:i + batch_size]
            pca_transformed_coords = []
            for idx, coords in enumerate(batch_coords_list):
                transformed_coords = self.apply_pca(coords)
                pca_transformed_coords.append(transformed_coords)

            # Flatten the PCA-transformed coordinates and convert to a 2D array for partial fitting
            batch_coords = torch.cat([coords.view(-1, 3) for coords in pca_transformed_coords], dim=0)
            batch_coords_np = batch_coords.cpu().numpy()

            # Partially fit the normalizer on the batch
            self.normalizer.partial_fit(batch_coords_np)

    def apply_pca(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA transformation individually to standardize the rotation of the coordinates.

        Parameters:
        -----------
        coords : torch.Tensor
            A tensor of shape (N, 4, 3) representing the coordinates of a protein structure.

        Returns:
        --------
        torch.Tensor
            The PCA-transformed coordinates with the same shape as the input.
        """
        # Center the coordinates
        centered_coords = self.recenter_coords(coords)

        # Apply PCA
        pca = PCA(n_components=3)
        pca.fit(centered_coords.view(-1, 3).cpu().numpy())

        # Ensure the PCA components form a proper rotation matrix (Determinant = 1)
        rotation_matrix = pca.components_.T
        # Convert it to fp32 for numerical stability
        rotation_matrix = rotation_matrix.astype(np.float32)
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, -1] = -rotation_matrix[:, -1]

        # Correct orthonormality using SVD
        u, _, vh = np.linalg.svd(rotation_matrix)
        corrected_rotation_matrix = np.dot(u, vh)

        # Apply the rotation matrix to the centered coordinates
        transformed_coords_np = centered_coords.cpu().numpy().dot(corrected_rotation_matrix)

        # Convert back to tensor and reshape to original shape
        transformed_coords = torch.tensor(transformed_coords_np, dtype=coords.dtype, device=coords.device)

        return transformed_coords

    @staticmethod
    def recenter_coords(coords: torch.Tensor) -> torch.Tensor:
        """
        Recenter the coordinates of a protein structure to its geometric center.

        Parameters:
        -----------
        coords : torch.Tensor
            A tensor of shape (N, 3) representing the coordinates of a protein structure.

        Returns:
        --------
        torch.Tensor
            The recentered coordinates.
        """
        # Get the geometric center of the coordinates
        center = torch.mean(coords, dim=0, keepdim=True)
        # Subtract the center from the coordinates
        recentered_coords = coords - center
        return recentered_coords

    def normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize the individually PCA-transformed coordinates using the fitted normalizer.

        Parameters:
        -----------
        coords : torch.Tensor
            A tensor of shape (N, 4, 3) representing the coordinates of a protein structure.

        Returns:
        --------
        torch.Tensor
            The normalized coordinates with the same shape as the input.
        """
        if self.normalizer is None:
            raise ValueError("Normalizer has not been fitted. Please call fit_normalizer() first "
                             "or load a saved one using load_normalizer.")

        # Apply individual PCA transformation
        coords_pca = self.apply_pca(coords)

        # Flatten the PCA-transformed coordinates and convert to a 2D array for normalization
        coords_np = coords_pca.view(-1, 3).cpu().numpy()

        # Normalize the coordinates
        normalized_coords_np = self.normalizer.transform(coords_np)

        # Convert back to tensor and reshape to original shape
        normalized_coords = torch.tensor(normalized_coords_np, dtype=coords.dtype, device=coords.device)
        normalized_coords = normalized_coords.view(coords.shape)

        return normalized_coords

    def denormalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the coordinates using the fitted normalizer.

        Parameters:
        -----------
        coords : torch.Tensor
            A tensor of shape (N, 4, 3) representing the normalized coordinates of a protein structure.

        Returns:
        --------
        torch.Tensor
            The denormalized coordinates with the same shape as the input.
        """
        if self.normalizer is None:
            raise ValueError("Normalizer has not been fitted. Please call fit_normalizer() first.")

        # Flatten the coordinates and convert to a 2D array for denormalization
        coords_np = coords.view(-1, 3).detach().cpu().numpy()

        # Denormalize the coordinates
        denormalized_coords_np = self.normalizer.inverse_transform(coords_np)

        # Convert back to tensor and reshape to original shape
        denormalized_coords_np = torch.tensor(denormalized_coords_np, dtype=coords.dtype, device=coords.device)
        denormalized_coords_np = denormalized_coords_np.view(coords.shape)

        return denormalized_coords_np

    def save_normalizer(self, file_path: str):
        """
        Save the fitted normalizer to a file.

        Parameters:
        -----------
        file_path : str
            The file path to save the normalizer.
        """
        joblib.dump({'normalizer': self.normalizer}, file_path)
        print(f"Normalizer saved to {file_path}")

    def load_normalizer(self, file_path: str):
        """
        Load the fitted normalizer from a file.

        Parameters:
        -----------
        file_path : str
            The file path from which to load the normalizer.
        """
        model_dict = joblib.load(file_path)
        self.normalizer = model_dict['normalizer']


def fit_normalizer():
    import yaml
    import tqdm
    from utils.utils import load_configs
    from torch.utils.data import DataLoader

    config_path = "../configs/config_gvp.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    dataset = NormalizerDataset(test_configs.train_settings.data_path, configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=1, num_workers=16, pin_memory=True)

    coords_list = []
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        coords_list.append(batch.squeeze(0).to(torch.float32))

    # Create an instance of the class
    processor = Protein3DProcessing()

    # Fit the PCA and normalizer on the dataset
    processor.fit_normalizer(coords_list)

    # Save the normalizer to a file for future use
    processor.save_normalizer("normalizer.pkl")

    # Load the normalizer from the file
    processor.load_normalizer("normalizer.pkl")

    # Normalize new coordinates using the loaded normalizer and PCA model
    new_coords = torch.rand((12, 4, 3))  # Example new protein structure
    normalized_coords = processor.normalize_coords(new_coords)

    # Denormalize the coordinates to get back the original values
    denormalized_coords = processor.denormalize_coords(normalized_coords)

    print(denormalized_coords)
    print("Normalizer fitted successfully!")


if __name__ == '__main__':
    fit_normalizer()
