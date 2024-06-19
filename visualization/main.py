import os.path
import numpy as np
from tqdm import tqdm
import torch
import glob
from visualization.tsne_plot import compute_plot
import yaml
import h5py
from utils.utils import load_configs, get_dummy_logger
from accelerate import Accelerator
from models.vqvae import prepare_models_vqvae
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from torch.utils.data import DataLoader, Dataset
import os


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


def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        plddt_scores = f['plddt_scores'][:]
    return seq, n_ca_c_o_coord, plddt_scores


def load_checkpoints(net, resume_path):
    # Load the checkpoint
    checkpoint = torch.load(resume_path, map_location='cpu')

    # Get the state_dict from the checkpoint
    pretrained_state_dict = checkpoint['model_state_dict']

    # Load the state_dict into the net
    log = net.load_state_dict(pretrained_state_dict)
    print(log)
    return net


class VQVAEDataset(Dataset):
    def __init__(self, data_path, rotate_randomly=True, **kwargs):
        super(VQVAEDataset, self).__init__()

        self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))[:kwargs['configs'].train_settings.max_task_samples]

        self.max_length = kwargs['configs'].model.max_length
        self.rotate_randomly = rotate_randomly

        self.processor = Protein3DProcessing()

        # Load saved pca and scaler models for processing
        self.processor.load_normalizer(kwargs['configs'].normalizer_path)

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

            for i in range(0, coords.shape[0]-1):
                if nan_mask[i].any() and not torch.isnan(coords[i + 1]).any():
                    coords[i] = coords[i + 1]

        return coords.view(original_shape)

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        sample = load_h5_file(sample_path)
        basename = os.path.basename(sample_path)
        pid = basename.split('.h5')[0].split('_')[0]
        coords_list = sample[1].tolist()
        coords_tensor = torch.Tensor(coords_list)

        coords_tensor = coords_tensor[:self.max_length, ...]

        coords_tensor = self.handle_nan_coordinates(coords_tensor)
        coords_tensor = self.processor.normalize_coords(coords_tensor)

        # Merge the features and create a mask
        coords_tensor = coords_tensor.reshape(1, -1, 12)
        # coords, masks = merge_features_and_create_mask(coords_tensor, self.max_length)

        # squeeze coords and masks to return them to 2D
        coords = coords_tensor.squeeze(0)
        # masks = masks.squeeze(0)

        return {'pid': pid, 'input_coords': coords}


def compute_visualization(net, test_loader, result_path, configs, logging, accelerator, epoch):
    plot_save_path = os.path.join(result_path, "tsne_plot")
    npz_path = os.path.join(result_path, "sequence_representations.npz")

    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    net.eval()
    representations = {'ids': [], 'rep': []}
    for batch in tqdm(test_loader, total=len(test_loader), desc="Computing representations", leave=False,
                      disable=not configs.tqdm_progress_bar):
        batch['input_coords'] = batch['target_coords']
        pid = batch['pid']
        with torch.inference_mode():
            x, *_ = net(batch, return_vq_only=True)
            x = x.cpu()
            if len(x.shape) == 3:
                output = x.permute(0, 2, 1).squeeze()
                output = output[batch['masks'].squeeze().cpu()]
            else:
                output = x.squeeze()
                output = output[batch['masks'].squeeze().cpu()]
                output = output.reshape(-1, output.shape[1])

            output = output.mean(dim=0)
            representations['ids'].append(pid[0])
            representations['rep'].append(output.numpy())

    np.savez_compressed(npz_path, **representations)

    compute_plot(fasta_file=configs.visualization_settings.fasta_path, npz_file=npz_path,
                 save_path=plot_save_path, epoch=epoch)


def main():
    config_path = "./../results/test/2024-06-13__17-32-42/config_vqvae.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()

    fasta_path = 'Rep_subfamily_basedon_S40pdb.fa'
    npz_path = "sequence_representations.npz"
    plot_save_path = "./plots/test_1_1x_4"
    checkpoint_path = "./../results/test/2024-06-13__17-32-42/checkpoints/best_valid.pth"

    dataset = VQVAEDataset(test_configs.valid_settings.data_path, configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)
    accelerator = Accelerator(
        mixed_precision="fp16",
    )

    net = prepare_models_vqvae(test_configs, test_logger, accelerator)

    net = load_checkpoints(net, checkpoint_path)

    net, test_loader = accelerator.prepare(net, test_loader)
    net.to(accelerator.device)

    net.eval()
    representations = {'ids': [], 'rep': []}
    for batch in tqdm(test_loader, total=len(test_loader), desc="Computing representations"):
        pid = batch['pid']
        with torch.inference_mode():
            x, indices, commit_loss = net(batch, return_vq_only=True)
            x = x.cpu()
            output = x.permute(0, 2, 1).squeeze()
            output = output.mean(dim=0)
            representations['ids'].append(pid[0])
            representations['rep'].append(output.numpy())

    np.savez_compressed(npz_path, **representations)

    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    compute_plot(fasta_file=fasta_path, npz_file=npz_path,
                 save_path=plot_save_path)


if __name__ == '__main__':
    main()
