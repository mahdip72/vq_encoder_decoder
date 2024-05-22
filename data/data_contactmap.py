import pcmap
import pypstruct
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


class ContactMapDataset(Dataset):
    """
    Dataset for protein contact map
    """
    def __init__(self, pdb_file, transform=None):
        self.contactmap = pdb_to_cmap(pdb_file)

    def __len__(self):
        return len(self.contactmap)

    def __getitem__(self, idx):
        return self.contactmap[idx]


def load_cmap_data(pdb_file, batch_size, shuffle):
    """
    Get a contact map data loader for the given PDB file.
    """
    cmap_dataset = ContactMapDataset(pdb_file)
    data_loader = DataLoader(dataset=cmap_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


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


def pdb_to_cmap(pdb_file):
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
        root_id = int(root_dict["root"]["resID"])
        for partner_dict in root_dict["partners"]:
            partner_id = int(partner_dict["resID"])
            cmap_matrix[root_id - 1][partner_id - 1] = 1
            cmap_matrix[partner_id - 1][root_id - 1] = 1

    return cmap_matrix


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


pdb_file = "../../7xhg_native_clean.pdb"

cmap0 = pdb_to_cmap(pdb_file)
print(len(cmap0))
print(cmap0)
fig, ax = plt.subplots()
plot_contact_map(cmap0, ax)
plt.show()