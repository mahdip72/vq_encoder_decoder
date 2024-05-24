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
    Dataset for protein contact map
    """
    def __init__(self, pdb_dir, threshold=8, chain="A", transform=None):
        self.pdbs = list(Path(pdb_dir).glob("*.pdb"))
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
    :param pdb_dir: (string) path to PDB directory
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


def calc_dist_matrix(chain_one, chain_two):
    """
    Return a matrix of C-alpha distances between two protein chains.
    :param chain_one: first chain
    :param chain_two: second chain
    :return: (torch.tensor) distance matrix
    """

    # Extract C-alpha coordinates
    coords_one = np.array([residue["CA"].coord for residue in chain_one])
    coords_two = np.array([residue["CA"].coord for residue in chain_two])

    # Calculate pairwise distances using scipy.spatial.distance.cdist
    dist_matrix = distance.cdist(coords_one, coords_two, 'euclidean')

    # Convert the distance matrix to a PyTorch tensor
    answer = torch.tensor(dist_matrix, dtype=torch.float32)

    return answer


def pdb_to_cmap(id, pdb_file, threshold=8, chain="A"):
    """
    Convert a PDB file to a contact map. The contact map is a matrix
    such that element ij is 1 if the C-alpha distance between residues i and j
    is less than the threshold, and 0 otherwise.
    :param id: (string) ID of the protein structure
    :param pdb_file: (String) path to the PDB file
    :param threshold: (int) threshold distance for contacts
    :param chain: (string) name of the chain to consider
    :return: (torch.tensor) contact map
    """
    structure = Bio.PDB.PDBParser().get_structure(id, pdb_file)
    model = structure[0]
    dist_matrix = calc_dist_matrix(model[chain], model[chain])
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
    pdb_dir = "/media/mpngf/Samsung USB/PDB_files/Alphafold database/swissprot_pdb_v4/"
    #pdb_dir = "PDB_database"
    #pdb_dir = "../../data/swissprot_pdb_v4"
    dataloader = prepare_dataloaders(pdb_dir)
    #i = 0
    for cmap, pdb_file in tqdm.tqdm(dataloader, total=len(dataloader)):
        """
        if i  < 10:
            fig, ax = plt.subplots()
            plot_contact_map(cmap[0], ax, title=str(pdb_file[0]))
            plt.show()
        """
        #i += 1
        pass
