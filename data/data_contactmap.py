import pcmap
import pypstruct
import torch
import numpy as np

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
    cmap_matrix = torch.zeros(max_residue, max_residue)
    for root_dict in cmap["data"]:
        root_id = int(root_dict["root"]["resID"])
        for partner_dict in root_dict["partners"]:
            partner_id = int(partner_dict["resID"])
            cmap_matrix[root_id - 1][partner_id - 1] = 1
            cmap_matrix[partner_id - 1][root_id - 1] = 1

    return cmap_matrix