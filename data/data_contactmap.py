import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

import pcmap
import pypstruct
import Bio.PDB


class ContactMapDataset(Dataset):
    """
    Dataset for converting PDB or mmCIF protein files to contact maps.
    """
    def __init__(self, pdb_dir, threshold=8, chain="A"):
        """
        :param pdb_dir: (string or Path) path to directory of PDB or mmCIF files
        :param threshold: (int) threshold distance for contacting residues
        :param chain: (string) name of chain to consider for contacts
        """
        self.pdbs = list(Path(pdb_dir).glob("*.[pdb cif]*"))
        self.threshold = threshold
        self.chain = chain

    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb_file = str(self.pdbs[idx])
        #contactmap = pdb_to_cmap_old(pdb_file) # Use the old function
        contactmap = pdb_to_cmap(str(pdb_file), pdb_file, self.threshold, self.chain)
        return contactmap, pdb_file


def prepare_dataloaders(pdb_dir):
    """
    Get a contact map data loader for the given PDB directory.
    Batch size = 1 because different proteins may have different numbers of residues.
    :param pdb_dir: (string or Path) path to PDB directory
    """
    dataset = ContactMapDataset(pdb_dir)
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


def calc_dist_matrix(chain):
    """
    Return a matrix of C-alpha distances between the residues of a protein chain.
    :param chain: protein chain
    :return: (torch.tensor) distance matrix
    """

    # Extract C-alpha coordinates
    coords = []
    nan_indices = set()  # Indices of NaN coordinates
    for residue in chain:

        # Only consider amino acid residues (ignore HETATM, HOH, etc.)
        if residue.id[0] == " ":

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


def pdb_to_cmap(protein_id, pdb_file, threshold=8, chain="A"):
    """
    Construct a contact map from a PDB or mmCIF file. The contact map is a matrix
    such that element ij is 1 if the C-alpha distance between residues i and j
    is less than the threshold, and 0 otherwise.
    :param protein_id: (string) ID of the protein structure
    :param pdb_file: (String or Path) path to the PDB or mmCIF file
    :param threshold: (int) threshold distance for contacts
    :param chain: (string) name of the chain to consider
    :return: (torch.tensor) contact map
    """
    structure = Bio.PDB.Structure.Structure("")
    file_ext = str(pdb_file)[-4:]

    # Parse PDB file
    if file_ext == ".pdb":
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(protein_id, pdb_file)
    # Parse CIF file
    elif file_ext == ".cif":
        structure = Bio.PDB.MMCIFParser(QUIET=True).get_structure(protein_id, pdb_file)
    else:
        return None

    # Construct the contact map
    model = structure[0]
    dist_matrix = calc_dist_matrix(model[chain])
    contact_map = dist_matrix < threshold
    return contact_map.to(torch.uint8)


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
    import tqdm
    # Test dataloader on PDB directory
    #pdb_dir = "/media/mpngf/Samsung USB/PDB_files/Alphafold database/swissprot_pdb_v4/"
    #pdb_directory = "PDB_database"
    pdb_directory = "../../data/swissprot_pdb_v4"
    dataloader = prepare_dataloaders(pdb_directory)

    n = 0
    for cmap, pdb_filename in tqdm.tqdm(dataloader, total=len(dataloader)):
        print(str(pdb_filename))
        # Plot the contact maps
        """
        if n < 11:
            fig, ax = plt.subplots()
            plot_contact_map(cmap[0], ax, title=str(pdb_filename[0]))
            plt.show()
        
        """
        n += 1
        pass
