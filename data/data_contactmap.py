import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
import os
import glob

from utils.utils import load_configs
from utils.utils import load_h5_file
from data.dataset import Protein3DProcessing
from data.preprocess_pdb import check_chains, filter_best_chains

import pcmap
import pypstruct
import Bio.PDB


class ContactMapDataset(Dataset):
    """
    Dataset for converting PDB or mmCIF protein files to contact maps.
    """
    def __init__(self, pdb_dir, threshold=8):
        """
        :param pdb_dir: (string or Path) path to directory of PDB or mmCIF files
        :param threshold: (int) threshold distance for contacting residues
        """
        self.pdbs = list(Path(pdb_dir).glob("*.[pdb cif]*"))
        self.threshold = threshold

        self.dictn = {
            'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
            'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
            'ASX': 'B', 'GLX': 'Z', 'PYL': 'O', 'SEC': 'U',  # 'UNK': 'X'
        }

        self.report_dict = {'protein_complex': 0, 'no_chain_id_a': 0, 'h5_processed': 0,
                                    'single_amino_acid': 0, 'error': 0}

    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb_file = str(self.pdbs[idx])
        #contactmap = pdb_to_cmap_old(pdb_file) # Use the old function
        contactmaps = pdb_to_cmap(str(pdb_file), pdb_file, self.dictn, self.report_dict, self.threshold)
        # TODO: deal with multiple contact maps per pdb file (multiple chains)

        if len(contactmaps) == 0:
            # Return a placeholder value if there are no valid chains
            # TODO: This is not compatible with convolution model
            return torch.zeros(1,1), pdb_file
        else:
            # If there are multiple contact maps, only return the first one
            first_chain_id = next(iter(contactmaps))
            return contactmaps[first_chain_id], pdb_file


def merge_features_and_create_mask(features_list, max_length=512):
    # Pad tensors and create mask
    padded_tensors = []
    mask_tensors = []
    for t in features_list:
        if t.size(0) < max_length:
            size_diff = max_length - t.size(0)
            pad = torch.zeros(size_diff, t.size(1), device=t.device)
            t_padded = torch.cat([t, pad], dim=0)
            mask = torch.cat([torch.ones(t.size(0), dtype=torch.bool, device=t.device),
                              torch.zeros(size_diff, dtype=torch.bool, device=t.device)], dim=0)
        else:
            t_padded = t
            mask = torch.ones(t.size(0), dtype=torch.bool, device=t.device)
        padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation
        mask_tensors.append(mask.unsqueeze(0))  # Add an extra dimension for concatenation

    # Concatenate tensors and masks
    result = torch.cat(padded_tensors, dim=0)
    mask = torch.cat(mask_tensors, dim=0)
    return result, mask


class DistanceMapVQVAEDataset(Dataset):
    def __init__(self, data_path, train_mode=False, rotate_randomly=False, **kwargs):
        super(DistanceMapVQVAEDataset, self).__init__()

        self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))[:kwargs['configs'].train_settings.max_task_samples]

        self.max_length = kwargs['configs'].model.max_length

        self.train_mode = train_mode

        self.rotate_randomly = rotate_randomly
        self.cutout = kwargs['configs'].train_settings.cutout.enable
        self.min_mask_size = kwargs['configs'].train_settings.cutout.min_mask_size
        self.max_mask_size = kwargs['configs'].train_settings.cutout.max_mask_size
        self.max_cuts = kwargs['configs'].train_settings.cutout.max_cuts

        self.processor = Protein3DProcessing()

        # Load saved pca and scaler models for processing
        #self.processor.load_normalizer(kwargs['configs'].normalizer_path)

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
    def random_rotation_matrix():
        """
        Creates a random 3D rotation matrix.

        Returns:
            torch.Tensor: A 3x3 rotation matrix.
        """
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        psi = np.random.uniform(0, 2 * np.pi)

        r_x = torch.Tensor([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])

        r_y = torch.Tensor([[np.cos(phi), 0, np.sin(phi)],
                            [0, 1, 0],
                            [-np.sin(phi), 0, np.cos(phi)]])

        r_z = torch.Tensor([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1]])

        r = r_z @ r_y @ r_x

        return r

    def rotate_coords(self, coords):
        """
        Rotates the coordinates using a random rotation matrix.

        Parameters:
            coords (torch.Tensor): The coordinates to rotate.

        Returns:
            torch.Tensor: The rotated coordinates.
        """
        R = self.random_rotation_matrix()
        rotated_coords = torch.einsum('ij,nj->ni', R, coords.view(-1, 3)).reshape(coords.shape)

        # Ensure orthogonality
        assert torch.allclose(R.T @ R, torch.eye(3, dtype=torch.float32),
                              atol=1e-6), "Rotation matrix is not orthogonal"

        # Ensure proper rotation
        assert torch.isclose(torch.det(R),
                             torch.tensor(1.0, dtype=torch.float32)), "Rotation matrix determinant is not +1"

        return rotated_coords

    def cutout_augmentation(self, coords, min_mask_size, max_mask_size, max_cuts):
        """
        Apply cutout augmentation on the coordinates.

        Parameters:
        coords (torch.Tensor): The coordinates tensor to augment.
        min_mask_size (int): The minimum size of the mask to apply.
        max_mask_size (int): The maximum size of the mask to apply.
        max_cuts (int): The maximum number of cuts to apply.

        Returns:
        torch.Tensor: The augmented coordinates.
        """
        # Get the total length of the coordinates
        total_length = coords.shape[1]

        # Randomly select the number of cuts
        num_cuts = np.random.randint(1, max_cuts + 1)

        for _ in range(num_cuts):
            # Randomly select the size of the mask
            mask_size = np.random.randint(min_mask_size, max_mask_size + 1)

            # Randomly select the start index for the mask
            start_idx = torch.randint(0, total_length - mask_size + 1, (1,)).item()

            # Apply the mask
            coords[:, start_idx:start_idx + mask_size, :] = 0

        return coords

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

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        sample = load_h5_file(sample_path)
        basename = os.path.basename(sample_path)
        pid = basename.split('.h5')[0].split('_')[0]
        coords_list = sample[1].tolist()
        coords_tensor = torch.Tensor(coords_list)

        coords_tensor = coords_tensor[:self.max_length, ...]

        coords_tensor = self.handle_nan_coordinates(coords_tensor)
        #coords_tensor = self.processor.normalize_coords(coords_tensor)

        if self.rotate_randomly and self.train_mode:
            # Apply random rotation
            input_coords_tensor = self.rotate_coords(coords_tensor)
        else:
            input_coords_tensor = coords_tensor

        # Merge the features and create a mask
        coords_tensor = coords_tensor.reshape(1, -1, 12)
        input_coords_tensor = input_coords_tensor.reshape(1, -1, 12)
        if self.cutout and self.train_mode:
            input_coords_tensor = self.cutout_augmentation(input_coords_tensor, min_mask_size=self.min_mask_size,
                                                           max_mask_size=self.max_mask_size,
                                                           max_cuts=self.max_cuts)

        coords, masks = merge_features_and_create_mask(coords_tensor, self.max_length)
        input_coords_tensor, masks = merge_features_and_create_mask(input_coords_tensor, self.max_length)

        input_coords_tensor = input_coords_tensor[..., 3:6].reshape(1, -1, 3)
        coords = coords[..., 3:6].reshape(1, -1, 3)

        input_distance_map = self.create_distance_map(input_coords_tensor.squeeze(0))
        target_distance_map = self.create_distance_map(coords.squeeze(0))

        # input_coords_tensor = input_coords_tensor.reshape(-1, 12)
        # coords = coords.reshape(-1, 12)

        # expand the first dimension of distance maps
        input_distance_map = input_distance_map.unsqueeze(0)
        target_distance_map = target_distance_map.unsqueeze(0)
        return {'pid': pid, 'input_coords': input_coords_tensor.squeeze(0), 'input_distance_map': input_distance_map,
                'target_coords': coords.squeeze(0), 'target_distance_map': target_distance_map, 'masks': masks.squeeze(0)}


class ContactMapDataset(Dataset):
    """
    Dataset for converting PDB or mmCIF protein files to contact maps.
    """
    def __init__(self, data_path, configs, threshold=8):
        """
        :param pdb_dir: (string or Path) path to directory of PDB or mmCIF files
        :param threshold: (int) threshold distance for contacting residues
        """
        self.data_path = str(data_path)
        self.threshold = threshold

        self.dist_dataset = DistanceMapVQVAEDataset(self.data_path, configs=configs)

    def __len__(self):
        return len(self.dist_dataset)

    def __getitem__(self, idx):
        return dmap_to_cmap(self.dist_dataset[idx])


def prepare_dataloaders(configs):
    """
    Get a contact map data loader for the given PDB directory.
    Batch size = 1 because different proteins may have different numbers of residues.
    :param configs: configurations for contact map
    :return: data loader
    """
    prot_dir = configs.contact_map_settings.protein_dir
    threshold = configs.contact_map_settings.threshold
    dataset = ContactMapDataset(pdb_dir=prot_dir, threshold=threshold)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return data_loader, None


###########################################################
### OLD PDB_TO_CMAP FUNCTION (SLOW) ###
def get_max_residue(pdb_file):
    """
    Get the maximum residue number in a PDB file.
    :param pdb_file: (String) name of the PDB file
    :return: (int) maximum residue number
    """
    structure = pypstruct.parseFilePDB(pdb_file)
    res_list = np.array(structure.atomDictorize["seqRes"])
    res_list = res_list.astype(int)
    return np.max(res_list)


def pdb_to_cmap_old(pdb_file):
    """
    Convert a PDB file to a contact map.
    :param pdb_file: (String) name of the PDB file
    :return: (torch.tensor) contact map
    """

    # Get the maximum residue number of the PDB file
    max_residue = get_max_residue(pdb_file)

    # Get contact map as a dictionary
    cmap = pcmap.contactMap(pdb_file)

    # Convert contact map to a logical tensor
    cmap_matrix = torch.eye(max_residue)

    for root_dict in cmap["data"]:
        root_id = int(root_dict["root"]["resID"]) - 1
        for partner_dict in root_dict["partners"]:
            partner_id = int(partner_dict["resID"]) - 1
            cmap_matrix[root_id][partner_id] = 1
            cmap_matrix[partner_id][root_id] = 1

    return cmap_matrix
############################################################


def get_best_chains(structure, report_dict):
    """
    Get the best chains of a protein structure and update report_dict accordingly.
    :param structure: (Bio.PDB.Structure.Structure) protein structure
    :param report_dict: (dict) dictionary to keep track of certain metrics
    :return best_chains: (dict) dictionary of best chains
    """

    # Get the best chains from the protein structure
    chain_sequences = check_chains(structure, report_dict)
    best_chains = filter_best_chains(chain_sequences, structure)

    if len(best_chains) > 1:
        report_dict['protein_complex'] += 1
    if 'A' not in list(best_chains.keys()):
        report_dict['no_chain_id_a'] += 1

    return best_chains


def calc_avg_res_coord(residue):
    """
    Calculate the average coordinates of a residue.
    :param residue: (Bio.PDB.Residue) residue
    :return: (numpy.array) average coordinates
    """

    sum_atom_coords = np.array([0,0,0]) # Sum of atom coordinates
    count = 0 # Number of atoms with coordinates
    for atom in residue.get_atoms():
        if atom.coord is not None:
            sum_atom_coords = np.add(sum_atom_coords, atom.coord)
            count += 1

    if count > 0:
        return np.divide(sum_atom_coords, count)
    # Return NaN values if none of the atoms have coordinates
    else:
        return np.array([np.nan, np.nan, np.nan])


def fill_missing_coords(coords, nan_indices):
    """
    Fill missing coordinates in a list of coordinates by replacing them with
    the closest non-missing coordinate.
    :param coords: (list or array) list of coordinates
    :param nan_indices: (iterable) collection of indices of missing coordinates
    :return: list of coordinates with missing values removed
    """
    for index in nan_indices:
        i = index - 1
        j = index + 1
        num_coords = len(coords)

        # Replace current coordinate with nearest non-missing coordinate
        while i >= 0 and j < num_coords:
            if i >= 0:
                if np.nan not in coords[i]:
                    coords[index] = coords[i]
                    break
            elif j < num_coords:
                if np.nan not in coords[j]:
                    coords[index] = coords[j]
                    break
            i -= 1
            j += 1

    return coords


def calc_dist_matrix(chain, dictn, report_dict):
    """
    Return a matrix of C-alpha distances between the residues of a protein chain.
    :param chain: protein chain
    :param dictn: (dict) dictionary of amino acid codes
    :param report_dict: (dict) dictionary to keep track of certain metrics
    :return: (torch.tensor) distance matrix
    """

    # Extract C-alpha coordinates
    coords = []
    nan_indices = set()  # Indices of NaN coordinates
    for residue in chain:

        # Only consider amino acid residues (ignore HETATM, HOH, etc.)
        if residue.id[0] == " " and residue.resname in dictn:

            try:
                ca_coord = residue["CA"].coord
            # Use average residue coordinates if alpha carbon is missing
            except KeyError:
                ca_coord = calc_avg_res_coord(residue)

            # Use average residue coordinates if alpha carbon coordinates are missing
            if None in ca_coord:
                ca_coord = calc_avg_res_coord(residue)

            # Mark current index if ca_coord has missing values
            # (e.g. if all atoms in the residue are missing coordinates)
            if np.nan in ca_coord:
                nan_indices.add(len(coords))

            coords.append(ca_coord) # Add alpha carbon coordinates to list of coordinates

    coords = np.array(coords)
    coords = fill_missing_coords(coords, nan_indices)

    # Calculate pairwise distances using scipy.spatial.distance.cdist
    dist_matrix = distance.cdist(coords, coords, 'euclidean')

    # Convert the distance matrix to a PyTorch tensor
    answer = torch.tensor(dist_matrix, dtype=torch.float32)

    return answer


def dmap_to_cmap(dmap, threshold=8):
    """
    Convert a protein distance map to a contact map.
    :param dmap: (torch.Tensor) distance map
    :threshold: (int) threshold distance for contacts
    :return: (torch.Tensor) contact map
    """
    contact_map = dmap < threshold
    contact_map = contact_map.to(torch.float32)
    return contact_map


def pdb_to_cmap(protein_id, pdb_file, dictn, report_dict, threshold=8):
    """
    Construct a contact map from a PDB or mmCIF file. The contact map is a matrix
    such that element ij is 1 if the C-alpha distance between residues i and j
    is less than the threshold, and 0 otherwise.
    :param protein_id: (string) ID of the protein structure
    :param pdb_file: (String or Path) path to the PDB or mmCIF file
    :param threshold: (int) threshold distance for contacts
    :param dictn: (dict) dictionary of amino acid codes
    :param report_dict: (dict) dictionary to keep track of certain metrics
    :return: dict(str: torch.Tensor) dictionary with chain IDs as keys and contact maps as values
    """
    file_ext = str(pdb_file)[-4:]

    # Parse PDB file
    if file_ext == ".pdb":
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(protein_id, pdb_file)
    # Parse CIF file
    elif file_ext == ".cif":
        structure = Bio.PDB.MMCIFParser(QUIET=True).get_structure(protein_id, pdb_file)
    else:
        return None

    model = structure[0]

    best_chains = get_best_chains(structure, report_dict)

    contact_maps = {} # chain_id: contact_map
    # Iterate through each of the best chains
    for chain_id, sequence in best_chains.items():
        chain = model[chain_id]
        # Construct the contact map
        dist_matrix = calc_dist_matrix(chain, dictn, report_dict)
        contact_map = dist_matrix < threshold
        contact_map = contact_map.to(torch.uint8)
        contact_maps[chain_id] = contact_map

    return contact_maps


def plot_contact_map(contact_map, ax, title=""):
    """
    Plot a contact map
    :param contact_map: (torch.tensor) contact map
    :param ax: (matplotlib axes) axes
    :param title: (String) title
    :return: None
    """
    ax.imshow(contact_map, cmap="binary")
    ax.set_xlabel("Residue Number")
    ax.set_ylabel("Residue Number")
    ax.set_title(title)


if __name__ == "__main__":
    # Test dataloader on PDB directory
    #pdb_dir = "/media/mpngf/Samsung USB/PDB_files/Alphafold database/swissprot_pdb_v4/"
    #pdb_directory = "PDB_database"
    #pdb_directory = "../../data/swissprot_pdb_v4"


    config_path = "../configs/config_vqvae_contact.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main_configs = load_configs(config_file)
    dataloader, placeholder = prepare_dataloaders(main_configs)

    n = 0
    for contactmap, pdb_filename in tqdm(dataloader, total=len(dataloader)):
        #print(str(pdb_filename))
        # Plot the contact maps
        #"""
        if n < 11:
            fig, axes = plt.subplots()
            plot_contact_map(contactmap[0], axes, title=str(pdb_filename[0]))
            plt.show()

        #"""
        n += 1
        pass
