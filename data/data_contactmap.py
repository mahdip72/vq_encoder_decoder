import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml
from utils.utils import load_configs
from preprocess_pdb import check_chains, filter_best_chains

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
            # TODO: is this appropriate?
            return [[0]], pdb_file
        else:
            # If there are multiple contact maps, only return the first one
            first_chain_id = next(iter(contactmaps))
            return contactmaps[first_chain_id], pdb_file


def prepare_dataloaders(configs):
    """
    Get a contact map data loader for the given PDB directory.
    Batch size = 1 because different proteins may have different numbers of residues.
    :param configs: configurations for contact map
    :return: data loader
    """
    pdb_dir = configs.contact_map_settings.protein_dir
    threshold = configs.contact_map_settings.threshold
    dataset = ContactMapDataset(pdb_dir=pdb_dir, threshold=threshold)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    return data_loader


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

    dataloader = prepare_dataloaders(main_configs)

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
