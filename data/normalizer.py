import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from utils.utils import load_h5_file
from torch.utils.data import DataLoader, Dataset
import os
import glob


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

        self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))[:kwargs['configs'].train_settings.max_task_samples]

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

    This class provides methods to fit PCA and normalization models, apply these transformations,
    and reverse the transformations (denormalize).

    Attributes:
    -----------
    pca : PCA
        The PCA model used for rotation standardization.
    normalizer : StandardScaler
        The normalizer used for coordinate normalization.

    Methods:
    --------
    fit_pca(coords_list)
        Fits the PCA model on the given list of coordinates.
    apply_pca(coords)
        Applies PCA transformation to standardize the rotation of the coordinates.
    fit_normalizer(coords_list)
        Fits the PCA and normalizer models on the given list of coordinates.
    normalize_coords(coords)
        Normalizes the PCA-transformed coordinates using the fitted normalizer.
    denormalize_coords(coords)
        Denormalizes the coordinates using the fitted normalizer and reverses PCA transformation.
    save_model(file_path)
        Saves the fitted PCA and normalizer models to a file.
    load_model(file_path)
        Loads the fitted PCA and normalizer models from a file.

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
    processor.save_model("model.pkl")
    processor.load_model("model.pkl")
    new_coords = torch.rand((12, 4, 3))  # Example new protein structure
    normalized_coords = processor.normalize_coords(new_coords)
    denormalized_coords = processor.denormalize_coords(normalized_coords)
    """

    def __init__(self, pca=None, normalizer=None):
        """
        Initialize the Protein3DProcessing class with optional PCA and normalizer models.

        Parameters:
        pca (PCA, optional): A fitted PCA model.
        normalizer (StandardScaler, optional): A fitted normalizer.
        """
        self.pca = pca
        self.normalizer = normalizer

    def fit_pca(self, coords_list: list):
        """
        Fit a PCA model on the given list of coordinates to standardize rotation.

        Parameters:
        coords_list (list): A list of torch.Tensors, each of shape (N, 4, 3) representing the coordinates
                            of multiple protein structures.
        """
        # Flatten the coordinates and convert to a 2D array for fitting
        all_coords = torch.cat([coords.view(-1, 3) for coords in coords_list], dim=0)
        all_coords_np = all_coords.cpu().numpy()

        # Create and fit the PCA model
        self.pca = PCA(n_components=3)
        self.pca.fit(all_coords_np)

    def apply_pca(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA transformation to standardize the rotation of the coordinates.

        Parameters:
        coords (torch.Tensor): A tensor of shape (N, 4, 3) representing the coordinates of a protein structure.

        Returns:
        torch.Tensor: The PCA-transformed coordinates with the same shape as the input.
        """
        if self.pca is None:
            raise ValueError("PCA model has not been fitted. Please call fit_pca() first.")

        # Flatten the coordinates and convert to a 2D array for transformation
        original_shape = coords.shape
        coords_np = coords.view(-1, 3).cpu().numpy()

        # Apply PCA transformation
        transformed_coords_np = self.pca.transform(coords_np)

        # Convert back to tensor and reshape to original shape
        transformed_coords = torch.tensor(transformed_coords_np, dtype=coords.dtype, device=coords.device)
        transformed_coords = transformed_coords.view(original_shape)

        return transformed_coords

    def fit_normalizer(self, coords_list: list):
        """
        Fit a normalizer on the given list of PCA-transformed coordinates.

        Parameters:
        coords_list (list): A list of torch.Tensors, each of shape (N, 4, 3) representing the coordinates
                            of multiple protein structures.
        """
        # Apply PCA to each set of coordinates
        self.fit_pca(coords_list)
        pca_transformed_coords = [self.apply_pca(coords) for coords in coords_list]

        # Flatten the PCA-transformed coordinates and convert to a 2D array for fitting
        all_coords = torch.cat([coords.view(-1, 3) for coords in pca_transformed_coords], dim=0)
        all_coords_np = all_coords.cpu().numpy()

        # Create and fit the normalizer
        self.normalizer = StandardScaler()
        self.normalizer.fit(all_coords_np)

    def normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize the PCA-transformed coordinates using the fitted normalizer.

        Parameters:
        coords (torch.Tensor): A tensor of shape (N, 4, 3) representing the coordinates of a protein structure.

        Returns:
        torch.Tensor: The normalized coordinates with the same shape as the input.
        """
        if self.normalizer is None:
            raise ValueError("Normalizer has not been fitted. Please call fit_normalizer() first.")

        # Apply PCA transformation
        coords_pca = self.apply_pca(coords)

        # Flatten the PCA-transformed coordinates and convert to a 2D array for normalization
        original_shape = coords_pca.shape
        coords_np = coords_pca.view(-1, 3).cpu().numpy()

        # Normalize the coordinates
        normalized_coords_np = self.normalizer.transform(coords_np)

        # Convert back to tensor and reshape to original shape
        normalized_coords = torch.tensor(normalized_coords_np, dtype=coords.dtype, device=coords.device)
        normalized_coords = normalized_coords.view(original_shape)

        return normalized_coords

    def denormalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the coordinates using the fitted normalizer and reverse PCA transformation.

        Parameters:
        coords (torch.Tensor): A tensor of shape (N, 4, 3) representing the normalized coordinates of a protein structure.

        Returns:
        torch.Tensor: The denormalized coordinates with the same shape as the input.
        """
        if self.normalizer is None or self.pca is None:
            raise ValueError("Normalizer and PCA model must be fitted. Please call fit_normalizer() first.")

        # Flatten the coordinates and convert to a 2D array for denormalization
        original_shape = coords.shape
        coords_np = coords.view(-1, 3).cpu().numpy()

        # Denormalize the coordinates
        denormalized_coords_np = self.normalizer.inverse_transform(coords_np)

        # Apply inverse PCA transformation
        original_coords_np = self.pca.inverse_transform(denormalized_coords_np)

        # Convert back to tensor and reshape to original shape
        original_coords = torch.tensor(original_coords_np, dtype=coords.dtype, device=coords.device)
        original_coords = original_coords.view(original_shape)

        return original_coords

    def save_model(self, file_path: str):
        """
        Save the fitted normalizer and PCA model to a file.

        Parameters:
        file_path (str): The file path to save the models.
        """
        joblib.dump({'normalizer': self.normalizer, 'pca': self.pca}, file_path)

    def load_model(self, file_path: str):
        """
        Load the fitted normalizer and PCA model from a file.

        Parameters:
        file_path (str): The file path from which to load the models.
        """
        model_dict = joblib.load(file_path)
        self.normalizer = model_dict['normalizer']
        self.pca = model_dict['pca']


def fit_normalizer():
    import yaml
    import tqdm
    from utils.utils import load_configs, get_dummy_logger
    from torch.utils.data import DataLoader
    from accelerate import Accelerator

    config_path = "../configs/config_gvp.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    dataset = NormalizerDataset(test_configs.train_settings.data_path, configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=1, num_workers=16, pin_memory=True)

    coords_list = []
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        coords_list.append(batch.squeeze(0))

    # Create an instance of the class
    processor = Protein3DProcessing()

    # Fit the PCA and normalizer on the dataset
    processor.fit_normalizer(coords_list)

    # Save the normalizer and PCA model to a file for future use
    processor.save_model("model.pkl")

    # Load the normalizer and PCA model from the file
    processor.load_model("model.pkl")

    # Normalize new coordinates using the loaded normalizer and PCA model
    new_coords = torch.rand((12, 4, 3))  # Example new protein structure
    normalized_coords = processor.normalize_coords(new_coords)

    # Denormalize the coordinates to get back the original values
    denormalized_coords = processor.denormalize_coords(normalized_coords)


if __name__ == '__main__':
    fit_normalizer()
