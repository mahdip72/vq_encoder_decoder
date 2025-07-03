import glob
import math
import numpy as np
import torch
import torch.nn.functional as F
import os
import functools
import random
from torch.utils.data import DataLoader, Dataset
from utils.utils import load_h5_file
from graphein.protein.resi_atoms import PROTEIN_ATOMS, STANDARD_AMINO_ACIDS, STANDARD_AMINO_ACID_MAPPING_1_TO_3
from torch_geometric.data import Batch


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


def custom_collate(one_batch):
    # Unpack the batch
    torch_geometric_feature = [item[0] for item in one_batch]  # item[0] is for torch_geometric Data

    # Create a Batch object
    torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
    raw_seqs = [item[1] for item in one_batch]
    plddt_scores = [item[2] for item in one_batch]
    pids = [item[3] for item in one_batch]

    coords = torch.stack([item[4] for item in one_batch])
    masks = torch.stack([item[5] for item in one_batch])

    input_coordinates = torch.stack([item[6] for item in one_batch])
    inverse_folding_labels = torch.stack([item[7] for item in one_batch])

    plddt_scores = torch.cat(plddt_scores, dim=0)
    batched_data = {'graph': torch_geometric_batch, 'seq': raw_seqs, 'plddt': plddt_scores, 'pid': pids,
                    'target_coords': coords, 'masks': masks, "input_coordinates": input_coordinates,
                    "inverse_folding_labels": inverse_folding_labels}
    return batched_data


def custom_collate_pretrained_gcp(one_batch, featuriser=None, task_transform=None, fill_value: float = 1e-5):
    # Unpack the batch
    torch_geometric_feature = [item[0] for item in one_batch]  # item[0] is for torch_geometric Data

    # Create a Batch object
    torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
    raw_seqs = [item[1] for item in one_batch]
    plddt_scores = [item[2] for item in one_batch]
    pids = [item[3] for item in one_batch]

    coords = torch.stack([item[4] for item in one_batch])
    masks = torch.stack([item[5] for item in one_batch])

    input_coordinates = torch.stack([item[6] for item in one_batch])
    inverse_folding_labels = torch.stack([item[7] for item in one_batch])

    plddt_scores = torch.cat(plddt_scores, dim=0)
    one_batch = {'graph': torch_geometric_batch, 'seq': raw_seqs, 'plddt': plddt_scores, 'pid': pids,
                 'target_coords': coords, 'masks': masks, "input_coordinates": input_coordinates,
                 "inverse_folding_labels": inverse_folding_labels}

    # Example ProteinWorkshopBatch(fill_value=[32], atom_list=[32], id=[32], residue_id=[32], residue_type=[10606], residues=[32], chains=[10606], coords=[10606, 37, 3], x=[10606], seq_pos=[10606, 1], one_batch=[10606], ptr=[33]):
    # fill_value: tensor([1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05, 1.0000e-05])
    # atom_list: ['N', 'CA', 'C', 'O', 'CB', 'OG', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OD1', 'ND2', 'CG1', 'CG2', 'CD', 'CE', 'NZ', 'OD2', 'OE1', 'NE2', 'OE2', 'OH', 'NE', 'NH1', 'NH2', 'OG1', 'SD', 'ND1', 'SG', 'NE1', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT']
    # id: ['1a0q', '3eiy', '1hcn', '4hhb', '1hcn', '4hhb', '1hcn', '4hhb', '1a0q', '1a0q', '1hcn', '3eiy', '1a0q', '1hcn', '1a0q', '1a0q', '1a0q', '4hhb', '1a0q', '3eiy', '4hhb', '1a0q', '4hhb', '3eiy', '3eiy', '3eiy', '3eiy', '4hhb', '3eiy', '3eiy', '3eiy', '3eiy']
    # residue_id: ['A:SER:2', 'A:PHE:3', 'A:SER:4', 'A:ASN:5', 'A:VAL:6', 'A:PRO:7', 'A:ALA:8', 'A:GLY:9', 'A:LYS:10', 'A:ASP:11', 'A:LEU:12', 'A:PRO:13', 'A:GLN:14', 'A:ASP:15', 'A:PHE:16', 'A:ASN:17', 'A:VAL:18', 'A:ILE:19', 'A:ILE:20', 'A:GLU:21', 'A:ILE:22', 'A:PRO:23', 'A:ALA:24', 'A:GLN:25', 'A:SER:26', 'A:GLU:27', 'A:PRO:28', 'A:VAL:29', 'A:LYS:30', 'A:TYR:31', 'A:GLU:32', 'A:ALA:33', 'A:ASP:34', 'A:LYS:35', 'A:ALA:36', 'A:LEU:37', 'A:GLY:38', 'A:LEU:39', 'A:LEU:40', 'A:VAL:41', 'A:VAL:42', 'A:ASP:43', 'A:ARG:44', 'A:PHE:45', 'A:ILE:46', 'A:GLY:47', 'A:THR:48', 'A:GLY:49', 'A:MET:50', 'A:ARG:51', 'A:TYR:52', 'A:PRO:53', 'A:VAL:54', 'A:ASN:55', 'A:TYR:56', 'A:GLY:57', 'A:PHE:58', 'A:ILE:59', 'A:PRO:60', ...]
    # residue_type: tensor([19, 15, 11,  ...,  5, 10, 10])
    # residues: ['SER', 'PHE', 'SER', 'ASN', 'VAL', 'PRO', 'ALA', 'GLY', 'LYS', 'ASP', 'LEU', 'PRO', 'GLN', 'ASP', 'PHE', 'ASN', 'VAL', 'ILE', 'ILE', 'GLU', 'ILE', 'PRO', 'ALA', 'GLN', 'SER', 'GLU', 'PRO', 'VAL', 'LYS', 'TYR', 'GLU', 'ALA', 'ASP', 'LYS', 'ALA', 'LEU', 'GLY', 'LEU', 'LEU', 'VAL', 'VAL', 'ASP', 'ARG', 'PHE', 'ILE', 'GLY', 'THR', 'GLY', 'MET', 'ARG', 'TYR', 'PRO', 'VAL', 'ASN', 'TYR', 'GLY', 'PHE', 'ILE', 'PRO', ...]
    # chains: tensor([0, 0, 0,  ..., 0, 0, 0])
    # coords: [ 1.0000e-05,  1.0000e-05,  1.0000e-05], [ 1.0000e-05,  1.0000e-05,  1.0000e-05], [ 1.0451e+01,  3.5432e+01, -1.0086e+01]]])
    # seq_pos: [[  0], [  1], [  2], ..., [171], [172], [173]])
    # one_batch: tensor([ 0,  0,  0,  ..., 31, 31, 31])
    # ptr: tensor([    0,   411,   585,   780,  1354,  1549,  2123,  2318,  2892,  3303, 3714,  3909,  4083,  4494,  4689,  5100,  5511,  5922,  6496,  6907, 7081,  7655,  8066,  8640,  8814,  8988,  9162,  9336,  9910, 10084, 10258, 10432, 10606])

    # build input graph one_batch to be featurized
    device = one_batch["graph"].x.device

    one_batch["graph"].fill_value = torch.full((one_batch["graph"].num_graphs,), fill_value, device=device)

    one_batch["graph"].atom_list = [PROTEIN_ATOMS for _ in range(one_batch["graph"].num_graphs)]
    one_batch["graph"].id = one_batch["pid"]

    one_batch["graph"].residue_id = [
        [f"A:{STANDARD_AMINO_ACID_MAPPING_1_TO_3[res]}:{res_index}" for res_index, res in enumerate(seq, start=1)] for
        seq in one_batch["seq"]]  # NOTE: this assumes all input graphs represent single-chain proteins
    one_batch["graph"].residue_type = torch.cat(
        [torch.tensor([STANDARD_AMINO_ACIDS.index(res) for res in seq], device=device) for seq in one_batch["seq"]])

    one_batch["graph"].residues = [[STANDARD_AMINO_ACID_MAPPING_1_TO_3[res] for res in seq] for seq in one_batch["seq"]]
    one_batch["graph"].chains = torch.zeros_like(one_batch["graph"].batch,
                                                 device=device)  # NOTE: this assumes all input graphs represent single-chain proteins

    one_batch["graph"].coords = torch.full((one_batch["graph"].num_nodes, len(PROTEIN_ATOMS), 3), fill_value,
                                           device=device, dtype=torch.float32)
    one_batch["graph"].coords[:, :3, :] = one_batch[
        "graph"].x_bb.float()  # NOTE: only the N, CA, and C atoms are referenced in the pretrained encoder, which requires float32 precision
    one_batch["graph"]._slice_dict["coords"] = one_batch["graph"]._slice_dict["x_bb"]

    one_batch["graph"].seq_pos = torch.cat([torch.arange(len(seq), device=device).unsqueeze(1) for seq in one_batch[
        "seq"]])  # NOTE: this assumes all input graphs represent single-chain proteins

    if featuriser:
        # Apply the featuriser to the collated graph batch
        one_batch['graph'] = featuriser(one_batch['graph'])
        # Apply the task transform if it exists
        if task_transform:
            one_batch['graph'] = task_transform(one_batch['graph'])
    return one_batch


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(d, d_min=0., d_max=20., d_count=16, device='cpu'):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `d` along a new axis=-1.
    That is, if `d` has shape [...dims], then the returned tensor will have
    shape [...dims, d_count].
    """
    d_mu = torch.linspace(d_min, d_max, d_count,
                          # device=device
                          )
    d_mu = d_mu.view([1, -1])
    d_sigma = (d_max - d_min) / d_count
    d_sigma = torch.tensor(d_sigma,
                           # device=device
                           )  # Convert d_sigma to a tensor
    d_expand = torch.unsqueeze(d, -1)

    RBF = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
    return RBF


def amino_acid_to_tensor(sequence, max_length):
    """
    Converts a single amino acid sequence to a categorical PyTorch tensor with padding or trimming.

    Args:
        sequence (str): The amino acid sequence (e.g., "ACDEFGHIKLMNPQRSTVWY").
        max_length (int): The desired fixed length for the sequence.

    Returns:
        torch.Tensor: A tensor of shape (1, max_length) representing the sequence.
    """
    # Define the amino acid vocabulary (20 standard + 4 non-standard types)
    amino_acids = "ACDEFGHIKLMNPQRSTVWYBZJX"  # B, Z, J, X are non-standard
    aa_to_index = {aa: i + 1 for i, aa in enumerate(amino_acids)}  # Map each AA to a unique index
    pad_index = 0  # padding index (0 for padding)

    # Convert the sequence to indices
    encoded = [aa_to_index.get(aa, pad_index) for aa in sequence]

    # Pad or trim the sequence to the desired length
    if len(encoded) < max_length:
        encoded = encoded + [pad_index] * (max_length - len(encoded))  # padding
    else:
        encoded = encoded[:max_length]  # trimming

    # Convert to a PyTorch tensor of shape (1, max_length)
    tensor = torch.tensor(encoded, dtype=torch.long)
    return tensor


class GCPNetDataset(Dataset):
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

    def __init__(self, data_path,
                 num_positional_embeddings=16, top_k=30, **kwargs
                 ):
        super(GCPNetDataset, self).__init__()
        self.h5_samples = glob.glob(os.path.join(data_path, '**', '*.h5'), recursive=True)

        self.mode = kwargs["mode"]

        if self.mode == 'train':
            random.shuffle(self.h5_samples)

        self.h5_samples = self.h5_samples[:kwargs['configs'].train_settings.max_task_samples]

        self.top_k = top_k
        self.num_positional_embeddings = num_positional_embeddings
        # self.node_counts = [len(e['seq']) for e in data_list]

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12, 'X': 20}
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

        self.max_length = kwargs['configs'].model.max_length

        self.configs = kwargs['configs']  # Main training config

        # Initialize attributes to None
        self.pretrained_featuriser = None
        self.pretrained_task_transform = None

        # Instantiate featuriser/transform only if using the pretrained GCPNet encoder
        if self.configs.model.encoder.name == "gcpnet" and self.configs.model.encoder.pretrained.enabled:
            # Import hydra and OmegaConf only when needed
            import hydra
            from omegaconf import OmegaConf

            pretrained_config_path = self.configs.model.encoder.pretrained.config_path
            # Load the configuration file associated with the pretrained model
            pretrained_cfg = OmegaConf.load(pretrained_config_path)

            # Instantiate the featuriser defined in the pretrained config
            self.pretrained_featuriser = hydra.utils.instantiate(pretrained_cfg.features)

            # Instantiate the task transform defined in the pretrained config (if it exists)
            if pretrained_cfg.get("task.transform"):
                self.pretrained_task_transform = hydra.utils.instantiate(pretrained_cfg.task.transform)

    @staticmethod
    def handle_nan_coordinates(coords: torch.Tensor) -> torch.Tensor:
        """
        Replaces NaN values in the coordinates with the previous or next valid coordinate values.

        Leading and trailing NaNs are filled using the nearest valid coordinate.
        If the entire tensor is NaN, all entries are set to zero.

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

    def __len__(self):
        return len(self.h5_samples)

    @staticmethod
    def recenter_coordinates(coordinates):
        """
        Recenters the 3D coordinates of a protein structure.

        Parameters:
        - coordinates: A PyTorch tensor of shape (num_amino_acids, 4, 3), representing
          the 3D coordinates of the backbone atoms.

        Returns:
        - A tensor of the same shape, but recentered to the center of the coordinates.
        """
        # Flatten to shape (num_atoms, 3) to calculate the center of mass
        num_amino_acids, num_atoms, _ = coordinates.shape
        flattened_coordinates = coordinates.view(-1, 3)

        # Compute the center of mass (mean of all coordinates)
        center_of_mass = flattened_coordinates.mean(dim=0)

        # Subtract the center of mass from each coordinate to recenter
        recentered_coordinates = coordinates - center_of_mass

        return recentered_coordinates

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        sample = load_h5_file(sample_path)
        basename = os.path.basename(sample_path)
        pid = basename.split('.h5')[0]

        # Decode sequence and replace U with X
        raw_sequence = sample[0].decode('utf-8').replace('U', 'X').replace('O', 'X')

        coords_list = self.recenter_coordinates(self.handle_nan_coordinates(torch.tensor(sample[1].tolist()))).tolist()
        # coords_list = torch.tensor(sample[1].tolist())
        sample_dict = {'name': pid,
                       'coords': coords_list,
                       'seq': raw_sequence}
        inverse_folding_labels = amino_acid_to_tensor(raw_sequence, self.max_length)

        feature = self._featurize_as_graph(sample_dict)
        plddt_scores = sample[2]
        plddt_scores = torch.from_numpy(plddt_scores).to(torch.float16) / 100
        raw_seqs = raw_sequence
        coords_list = sample_dict['coords']
        coords_tensor = torch.Tensor(coords_list)

        coords_tensor = coords_tensor[:self.max_length, ...]

        if self.configs.train_settings.cutoff_augmentation.enabled and self.mode == 'train':
            if random.random() < self.configs.train_settings.cutoff_augmentation.probability:
                # Randomly select a number between min_length and max_length
                random_length = random.randint(self.configs.train_settings.cutoff_augmentation.min_length,
                                               self.max_length)
                coords_tensor = coords_tensor[:random_length, ...]

        coords_tensor = coords_tensor.reshape(1, -1, 12)
        # Merge the features and create a mask
        coords, masks = merge_features_and_create_mask(coords_tensor, self.max_length)

        input_coordinates = coords.clone()
        coords = coords[..., :9]  # only use N, CA, C atoms

        # squeeze coords and masks to return them to 2D
        coords = coords.squeeze(0)
        masks = masks.squeeze(0)

        return [feature, raw_seqs, plddt_scores, pid, coords, masks, input_coordinates, inverse_folding_labels]

    def _featurize_as_graph(self, protein):
        import torch_cluster
        from torch_geometric.data import Data

        name = protein['name']
        with torch.no_grad():
            # x=N, C-alpha, C, and O atoms
            coords = torch.as_tensor(protein['coords'],
                                     # device=self.device,
                                     dtype=torch.float32)

            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                  # device=self.device,
                                  dtype=torch.long)

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            pos_embeddings = self._positional_embeddings(edge_index)

            dihedrals = self._dihedrals(coords)  # only this one used O
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            node_s = dihedrals

            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)

            edge_s = pos_embeddings  # NOTE: radial basis functions will be computed during the forward pass
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                                 (node_s, node_v, edge_s, edge_v))

        data = Data(x=X_ca, x_bb=coords[:, :3], seq=seq, name=name,
                    h=node_s, chi=node_v,
                    e=edge_s, xi=edge_v,
                    edge_index=edge_index, mask=mask)
        return data

    @staticmethod
    def _dihedrals(x, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        x = torch.reshape(x[:, :3], [3 * x.shape[0], 3])
        dx = x[1:] - x[:-1]
        U = _normalize(dx, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.linalg.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.linalg.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=(2, 1000)):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32,
                         # device=self.device
                         )
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    @staticmethod
    def _orientations(x):
        forward = _normalize(x[1:] - x[:-1])
        backward = _normalize(x[:-1] - x[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    @staticmethod
    def _sidechains(x):
        n, origin, c = x[:, 0], x[:, 1], x[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.linalg.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        # The specific weights used in the linear combination (math.sqrt(1 / 3) and math.sqrt(2 / 3)) are derived from the idealized tetrahedral geometry of the sidechain atoms around the C-alpha atom. However, in practice, these weights may be adjusted or learned from data to better capture the actual sidechain geometries observed in protein structures.
        return vec


def prepare_gcpnet_vqvae_dataloaders(logging, accelerator, configs, **kwargs):
    if accelerator.is_main_process:
        logging.info(f"train directory: {configs.train_settings.data_path}")
        logging.info(f"valid directory: {configs.valid_settings.data_path}")
        logging.info(f"visualization directory: {configs.visualization_settings.data_path}")

        assert os.path.exists(configs.train_settings.data_path), (
            f"Data path {configs.train_settings.data_path} does not exist"
        )
        assert os.path.exists(configs.valid_settings.data_path), (
            f"Data path {configs.valid_settings.data_path} does not exist"
        )
        assert os.path.exists(configs.visualization_settings.data_path), (
            f"Data path {configs.visualization_settings.data_path} does not exist"
        )

    train_dataset = GCPNetDataset(
        configs.train_settings.data_path,
        top_k=kwargs["encoder_configs"].top_k,
        num_positional_embeddings=kwargs["encoder_configs"].num_positional_embeddings,
        configs=configs,
        mode="train",
    )

    valid_dataset = GCPNetDataset(
        configs.valid_settings.data_path,
        top_k=kwargs["encoder_configs"].top_k,
        num_positional_embeddings=kwargs["encoder_configs"].num_positional_embeddings,
        configs=configs,
        mode="evaluation",
    )

    visualization_dataset = GCPNetDataset(
        configs.visualization_settings.data_path,
        top_k=kwargs["encoder_configs"].top_k,
        num_positional_embeddings=kwargs["encoder_configs"].num_positional_embeddings,
        configs=configs,
        mode="evaluation",
    )

    condition_met = configs.model.encoder.pretrained.enabled and configs.model.encoder.name == "gcpnet"
    custom_collate_pretrained_gcp_partial = functools.partial(
        custom_collate_pretrained_gcp,
        featuriser=train_dataset.pretrained_featuriser,
        task_transform=train_dataset.pretrained_task_transform
    )
    selected_collate = custom_collate_pretrained_gcp_partial if condition_met else custom_collate

    if condition_met:
        logging.info("Using custom collate function for GCPNet with pretrained encoder")
    else:
        logging.info("Using default collate function for GCPNet")

    train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                              num_workers=configs.train_settings.num_workers,
                              pin_memory=False,  # page-lock host buffers
                              persistent_workers=False,  # keep workers alive between epochs
                              collate_fn=selected_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size,
                              num_workers=configs.valid_settings.num_workers,
                              pin_memory=False,
                              persistent_workers=False,  # keep workers alive between epochs
                              shuffle=False,
                              collate_fn=selected_collate)
    visualization_loader = DataLoader(visualization_dataset, batch_size=configs.visualization_settings.batch_size,
                                      num_workers=0,
                                      pin_memory=False,
                                      shuffle=False,
                                      collate_fn=selected_collate)
    return train_loader, valid_loader, visualization_loader


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils.utils import load_configs, get_dummy_logger
    from torch.utils.data import DataLoader
    from accelerate import Accelerator

    # from utils.metrics import batch_distance_map_to_coordinates

    config_path = "../configs/config_vqvae.yaml"

    print('Loading config file:', config_path)
    with open(config_path) as file:
        config_file = yaml.full_load(file)

    config_file['model']['encoder']['pretrained'][
        'config_path'] = "../configs/pretrained/structure_denoising_pretrained_config.yaml"
    config_file['model']['encoder']['pretrained'][
        'checkpoint_path'] = "../models/checkpoints/structure_denoising/gcpnet/ca_bb/last.ckpt"
    test_configs = load_configs(config_file)

    test_configs.train_settings.data_path = '/home/mpngf/datasets/vqvae/swissprot_1024_h5/'
    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    print('data path:', os.path.abspath(test_configs.train_settings.data_path))
    dataset = GCPNetDataset(test_configs.train_settings.data_path, train_mode=True, rotate_randomly=False,
                            max_samples=test_configs.train_settings.max_task_samples,
                            configs=test_configs,
                            mode="train")

    # determine collate function based on pretrained GCPNet config
    condition_met = test_configs.model.encoder.pretrained.enabled and test_configs.model.encoder.name == "gcpnet"
    custom_collate_pretrained_gcp_partial = functools.partial(
        custom_collate_pretrained_gcp,
        featuriser=dataset.pretrained_featuriser,
        task_transform=dataset.pretrained_task_transform
    )
    selected_collate = custom_collate_pretrained_gcp_partial if condition_met else custom_collate

    test_loader = DataLoader(dataset, batch_size=test_configs.train_settings.batch_size,
                             num_workers=test_configs.train_settings.num_workers,
                             pin_memory=False,  # page-lock host buffers
                             persistent_workers=True,  # keep workers alive between epochs
                             prefetch_factor=2,
                             collate_fn=selected_collate)

    print('Building dataset with', len(dataset), 'samples')

    # test_loader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True)
    struct_embeddings = []
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        # graph = batch["graph"]
        pass
