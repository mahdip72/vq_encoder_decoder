import glob
import math
import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import DataLoader, Dataset
from utils.utils import load_h5_file
from sklearn.decomposition import PCA
from data.normalizer import Protein3DProcessing


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
    from torch_geometric.data import Batch

    # Unpack the batch
    torch_geometric_feature = [item[0] for item in one_batch]  # item[0] is for torch_geometric Data

    # Create a Batch object
    torch_geometric_batch = Batch.from_data_list(torch_geometric_feature)
    raw_seqs = [item[1] for item in one_batch]
    plddt_scores = [item[2] for item in one_batch]
    pids = [item[3] for item in one_batch]

    coords = torch.stack([item[4] for item in one_batch])
    masks = torch.stack([item[5] for item in one_batch])

    plddt_scores = torch.cat(plddt_scores, dim=0)
    batched_data = {'graph': torch_geometric_batch, 'seq': raw_seqs, 'plddt': plddt_scores, 'pid': pids,
                    'target_coords': coords, 'masks': masks}
    return batched_data


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


class GVPDataset(Dataset):
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
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16,
                 seq_mode="embedding", use_rotary_embeddings=False, rotary_mode=1,
                 use_foldseek=False, use_foldseek_vector=False, **kwargs
                 ):
        super(GVPDataset, self).__init__()
        from gvp.rotary_embedding import RotaryEmbedding
        if "cath_4_3_0" in data_path:
            self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))
        else:
            self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))[:kwargs['configs'].train_settings.max_task_samples]
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.seq_mode = seq_mode
        # self.node_counts = [len(e['seq']) for e in data_list]
        self.use_rotary_embeddings = use_rotary_embeddings
        self.rotary_mode = rotary_mode
        self.use_foldseek = use_foldseek
        self.use_foldseek_vector = use_foldseek_vector
        if self.use_rotary_embeddings:
            if self.rotary_mode == 3:
                self.rot_emb = RotaryEmbedding(dim=8)  # must be 5
            else:
                self.rot_emb = RotaryEmbedding(dim=2)  # must be 2

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

        self.letter_to_PhysicsPCA = {"C": [-0.132, 0.174, 0.070, 0.565, -0.374],
                                     "D": [0.303, -0.057, -0.014, 0.225, 0.156],
                                     "S": [0.199, 0.238, -0.015, -0.068, -0.196],
                                     "Q": [0.149, -0.184, -0.030, 0.035, -0.112],
                                     "K": [0.243, -0.339, -0.044, -0.325, -0.027],
                                     "I": [-0.353, 0.071, -0.088, -0.195, -0.107],
                                     "P": [0.173, 0.286, 0.407, -0.215, 0.384],
                                     "T": [0.068, 0.147, -0.015, -0.132, -0.274],
                                     "F": [-0.329, -0.023, 0.072, -0.002, 0.208],
                                     "A": [0.008, 0.134, -0.475, -0.039, 0.181],
                                     "G": [0.218, 0.562, -0.024, 0.018, 0.106],
                                     "H": [0.023, -0.177, 0.041, 0.280, -0.021],
                                     "E": [0.221, -0.280, -0.315, 0.157, 0.303],
                                     "L": [-0.267, 0.018, -0.265, -0.274, 0.206],
                                     "R": [0.171, -0.361, 0.107, -0.258, -0.364],
                                     "W": [-0.296, -0.186, 0.389, 0.083, 0.297],
                                     "V": [-0.274, 0.136, -0.187, -0.196, -0.299],
                                     "Y": [-0.141, -0.057, 0.425, -0.096, -0.091],
                                     "N": [0.255, 0.038, 0.117, 0.118, -0.055],
                                     "M": [-0.239, -0.141, -0.155, 0.321, 0.077]}

        # https://www.pnas.org/doi/full/10.1073/pnas.0408677102#sec-4
        self.letter_to_Atchleyfactor = {
            "A": [-0.591, -1.302, -0.733, 1.570, -0.146],
            "C": [-1.343, 0.465, -0.862, -1.020, -0.255],
            "D": [1.050, 0.302, -3.656, -0.259, -3.242],
            "E": [1.357, -1.453, 1.477, 0.113, -0.837],
            "F": [-1.006, -0.590, 1.891, -0.397, 0.412],
            "G": [-0.384, 1.652, 1.330, 1.045, 2.064],
            "H": [0.336, -0.417, -1.673, -1.474, -0.078],
            "I": [-1.239, -0.547, 2.131, 0.393, 0.816],
            "K": [1.831, -0.561, 0.533, -0.277, 1.648],
            "L": [-1.019, -0.987, -1.505, 1.266, -0.912],
            "M": [-0.663, -1.524, 2.219, -1.005, 1.212],
            "N": [0.945, 0.828, 1.299, -0.169, 0.933],
            "P": [0.189, 2.081, -1.628, 0.421, -1.392],
            "Q": [0.931, -0.179, -3.005, -0.503, -1.853],
            "R": [1.538, -0.055, 1.502, 0.440, 2.897],
            "S": [-0.228, 1.399, -4.760, 0.670, -2.647],
            "T": [-0.032, 0.326, 2.213, 0.908, 1.313],
            "V": [-1.337, -0.279, -0.544, 1.242, -1.262],
            "W": [-0.595, 0.009, 0.672, -2.128, -0.184],
            "Y": [0.260, 0.830, 3.097, -0.838, 1.512]}

        self.max_length = kwargs['configs'].model.max_length

        self.processor = Protein3DProcessing()

        # Load saved pca and scaler models for processing
        self.processor.load_normalizer(kwargs['configs'].normalizer_path)

    @staticmethod
    def normalize_coords(coords: torch.Tensor, divisor: int) -> torch.Tensor:
        """
        Normalize the coordinates of a protein structure by dividing by a fixed integer.

        Parameters:
        coords (torch.Tensor): A tensor of shape (N, 4, 3) where N is the number of amino acids,
                               and each amino acid has four 3D coordinates (x, y, z).
        divisor (int): The integer by which to divide all coordinates.

        Returns:
        torch.Tensor: The normalized coordinates with the same shape as the input.
        """
        if divisor == 0:
            raise ValueError("Divisor must be a non-zero integer")

        # Divide all coordinates by the specified integer
        normalized_coords = coords / divisor

        return normalized_coords

    @staticmethod
    def recenter_coords(coords: torch.Tensor) -> torch.Tensor:
        """
        Recenter the coordinates of a protein structure to its geometric center.
        The shape of coordinates are based on a list containing four coordinates
        for each amino acid in the protein structure:
        [[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)], ...].
        """
        # Reshape the tensor to 2D
        original_shape = coords.shape
        coords = coords.view(-1, 3)

        # Get the geometric center of the coordinates
        center = torch.mean(coords, dim=0, keepdim=True)

        # Subtract the center from the coordinates
        recentered_coords = coords - center

        # Reshape the tensor back to its original shape
        recentered_coords = recentered_coords.view(original_shape)

        return recentered_coords

    @staticmethod
    def align_coords(coords: torch.Tensor) -> torch.Tensor:
        """
        Align the coordinates of a protein structure using PCA.

        Parameters:
        coords (torch.Tensor): A tensor of shape (N, 4, 3) where N is the number of amino acids,
                               and each amino acid has four 3D coordinates (x, y, z).

        Returns:
        torch.Tensor: The aligned coordinates with the same shape as the input.
        """
        if coords.dim() != 3 or coords.size(1) != 4 or coords.size(2) != 3:
            raise ValueError("Input tensor must have the shape (N, 4, 3)")

        # Reshape the tensor to 2D (flatten the first two dimensions)
        original_shape = coords.shape
        coords = coords.view(-1, 3)

        # Perform PCA to find the principal components
        pca = PCA(n_components=3)
        pca.fit(coords.cpu().numpy())  # Ensure coords are on CPU for sklearn compatibility

        # Rotate the coordinates to align with the principal axes
        aligned_coords = pca.transform(coords.cpu().numpy())

        # Convert back to tensor and move to the original device
        aligned_coords = torch.tensor(aligned_coords, dtype=coords.dtype, device=coords.device)

        # Reshape the tensor back to its original shape
        aligned_coords = aligned_coords.view(original_shape)

        return aligned_coords

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

    def __len__(self):
        return len(self.h5_samples)

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        sample = load_h5_file(sample_path)
        basename = os.path.basename(sample_path)
        pid = basename.split('.h5')[0]
        sample_dict = {'name': pid,
                       'coords': sample[1].tolist(),
                       'seq': sample[0].decode('utf-8')}
        feature = self._featurize_as_graph(sample_dict)
        plddt_scores = sample[2]
        plddt_scores = torch.from_numpy(plddt_scores).to(torch.float16) / 100
        raw_seqs = sample[0].decode('utf-8')
        coords_list = sample[1].tolist()
        coords_tensor = torch.Tensor(coords_list)

        coords_tensor = coords_tensor[:self.max_length, ...]

        coords_tensor = self.handle_nan_coordinates(coords_tensor)
        coords_tensor = self.processor.normalize_coords(coords_tensor)

        # Recenter the coordinates center
        # coords_tensor = self.recenter_coords(coords_tensor)

        # Align the coordinates rotation
        # coords_tensor = self.align_coords(coords_tensor)

        # Normalize the coordinates
        # coords_tensor = self.normalize_coords(coords_tensor, 200)

        # Merge the features and create a mask
        coords_tensor = coords_tensor.reshape(1, -1, 12)
        coords, masks = merge_features_and_create_mask(coords_tensor, self.max_length)

        # squeeze coords and masks to return them to 2D
        coords = coords.squeeze(0)
        masks = masks.squeeze(0)

        return [feature, raw_seqs, plddt_scores, pid, coords, masks]

    def _featurize_as_graph(self, protein):
        import torch_cluster
        from torch_geometric.data import Data

        name = protein['name']
        with torch.no_grad():
            # x=N, C-alpha, C, and O atoms
            coords = torch.as_tensor(protein['coords'],
                                     # device=self.device,
                                     dtype=torch.float32)
            # print(self.seq_mode)
            if self.seq_mode == "PhysicsPCA":
                seq = torch.as_tensor([self.letter_to_PhysicsPCA[a] for a in protein['seq']],
                                      # device=self.device,
                                      dtype=torch.long)
            elif self.seq_mode == "Atchleyfactor":
                seq = torch.as_tensor([self.letter_to_Atchleyfactor[a] for a in protein['seq']],
                                      # device=self.device,
                                      dtype=torch.long)
            else:
                seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                      # device=self.device,
                                      dtype=torch.long)

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            if self.use_rotary_embeddings:
                if self.rotary_mode == 1:  # first mode
                    d1 = edge_index[0] - edge_index[1]
                    d2 = edge_index[1] - edge_index[0]
                    d = torch.cat((d1.unsqueeze(-1), d2.unsqueeze(-1)), dim=-1)  # [len,2]
                    pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
                if self.rotary_mode == 2:
                    d = edge_index.transpose(0, 1)
                    pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
                if self.rotary_mode == 3:
                    d = edge_index.transpose(0, 1)  # [len,2]
                    d = torch.cat((d, E_vectors, -1 * E_vectors), dim=-1)  # [len,2+3]
                    pos_embeddings = self.rot_emb(d.unsqueeze(0).unsqueeze(-2)).squeeze(0).squeeze(-2)
            else:
                pos_embeddings = self._positional_embeddings(edge_index)

            rbf = _rbf(E_vectors.norm(dim=-1), d_count=self.num_rbf,
                       # device=self.device
                       )

            dihedrals = self._dihedrals(coords)  # only this one used O
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            if self.use_foldseek or self.use_foldseek_vector:
                foldseek_features = self._foldseek(X_ca)

            if self.use_foldseek:
                node_s = torch.cat([dihedrals, foldseek_features[0]], dim=-1)  # add additional 10 features
            else:
                node_s = dihedrals

            if self.use_foldseek_vector:
                node_v = torch.cat([orientations, sidechains.unsqueeze(-2), foldseek_features[1]],
                                   dim=-2)  # add additional 18 features
            else:
                node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)

            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                                 (node_s, node_v, edge_s, edge_v))

        data = Data(x=X_ca, seq=seq, name=name,
                    node_s=node_s, node_v=node_v,
                    edge_s=edge_s, edge_v=edge_v,
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

    @staticmethod
    def _foldseek(x):
        # From Fast and accurate protein structure search with Foldseek
        # x is X_ca coordinates
        import torch_cluster

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        j, i = torch_cluster.knn_graph(x, k=1)
        mask = torch.zeros_like(i, dtype=torch.bool)
        first_unique = {}
        for index, value in enumerate(i):
            if value.item() not in first_unique:
                first_unique[value.item()] = index
                mask[index] = True
            else:
                mask[index] = False

        i, j = i[mask], j[mask]
        pad_x = F.pad(x, [0, 0, 1, 1])  # padding to the first and last
        j = j + 1
        i = i + 1
        d = torch.norm(pad_x[i] - pad_x[j], dim=-1, keepdim=True)
        abs_diff = torch.abs(i - j)
        sign = torch.sign(i - j)
        f1 = (sign * torch.minimum(abs_diff, torch.tensor(4))).unsqueeze(-1)
        f2 = (sign * torch.log(abs_diff + 1)).unsqueeze(-1)

        u1 = _normalize(pad_x[i] - pad_x[i - 1])
        u2 = _normalize(pad_x[i + 1] - pad_x[i])
        u3 = _normalize(pad_x[j] - pad_x[j - 1])
        u4 = _normalize(pad_x[j + 1] - pad_x[j])
        u5 = _normalize(pad_x[j] - pad_x[i])
        cos_12 = torch.sum(u1 * u2, dim=-1, keepdim=True)
        cos_34 = torch.sum(u3 * u4, dim=-1, keepdim=True)
        cos_15 = torch.sum(u1 * u5, dim=-1, keepdim=True)
        cos_35 = torch.sum(u3 * u5, dim=-1, keepdim=True)
        cos_14 = torch.sum(u1 * u4, dim=-1, keepdim=True)
        cos_23 = torch.sum(u2 * u3, dim=-1, keepdim=True)
        cos_13 = torch.sum(u1 * u3, dim=-1, keepdim=True)
        node_s_features = torch.cat([cos_12, cos_34, cos_15, cos_35, cos_14, cos_23, cos_13, d, f1, f2], dim=-1)
        node_v_features = torch.cat([u1.unsqueeze(-2), u2.unsqueeze(-2), u3.unsqueeze(-2),
                                     u4.unsqueeze(-2), u5.unsqueeze(-2), (pad_x[i] - pad_x[j]).unsqueeze(-2)],
                                    dim=-2)  # add 6 additional features
        return node_s_features, node_v_features


class VQVAEDataset(Dataset):
    def __init__(self, data_path, train_mode=False, rotate_randomly=False, **kwargs):
        super(VQVAEDataset, self).__init__()

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
            mask_size = np.random.randint(min_mask_size, min(max_mask_size + 1, total_length))

            # Randomly select the start index for the mask
            start_idx = torch.randint(0, total_length - mask_size + 1, (1,)).item()

            # Apply the mask
            coords[:, start_idx:start_idx + mask_size, :] = 0

        return coords

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

        # squeeze coords and masks to return them to 2D
        coords = coords.squeeze(0)
        input_coords_tensor = input_coords_tensor.squeeze(0)
        masks = masks.squeeze(0)

        return {'pid': pid, 'input_coords': input_coords_tensor, 'target_coords': coords, 'masks': masks}


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
        # coords_tensor = self.processor.normalize_coords(coords_tensor)

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

        coords_tensor, masks = merge_features_and_create_mask(coords_tensor, self.max_length)
        input_coords_tensor, masks = merge_features_and_create_mask(input_coords_tensor, self.max_length)

        input_coords_tensor = input_coords_tensor[..., 3:6].reshape(1, -1, 3)
        coords_tensor = coords_tensor[..., 3:6].reshape(1, -1, 3)

        input_distance_map = self.create_distance_map(input_coords_tensor.squeeze(0))
        target_distance_map = self.create_distance_map(coords_tensor.squeeze(0))

        input_distance_map = self.processor.normalize_distance_map(input_distance_map)
        target_distance_map = self.processor.normalize_distance_map(target_distance_map)

        # input_coords_tensor = input_coords_tensor.reshape(-1, 12)
        # coords = coords.reshape(-1, 12)

        # expand the first dimension of distance maps
        input_distance_map = input_distance_map.unsqueeze(0)
        target_distance_map = target_distance_map.unsqueeze(0)
        return {'pid': pid, 'input_coords': input_coords_tensor.squeeze(0), 'input_distance_map': input_distance_map,
                'target_coords': coords_tensor.squeeze(0), 'target_distance_map': target_distance_map, 'masks': masks.squeeze(0)}


def prepare_gvp_vqvae_dataloaders(logging, accelerator, configs):
    if accelerator.is_main_process:
        logging.info(f"train directory: {configs.train_settings.data_path}")
        logging.info(f"valid directory: {configs.valid_settings.data_path}")
        logging.info(f"visualization directory: {configs.visualization_settings.data_path}")

    if hasattr(configs.model.struct_encoder, "use_seq") and configs.model.struct_encoder.use_seq.enable:
        seq_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
    else:
        seq_mode = "embedding"

    train_dataset = GVPDataset(
        configs.train_settings.data_path,
        seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode,
        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
        use_foldseek=configs.model.struct_encoder.use_foldseek,
        use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
        top_k=configs.model.struct_encoder.top_k,
        num_rbf=configs.model.struct_encoder.num_rbf,
        num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings,
        configs=configs
    )

    valid_dataset = GVPDataset(
        configs.valid_settings.data_path,
        seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode,
        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
        use_foldseek=configs.model.struct_encoder.use_foldseek,
        use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
        top_k=configs.model.struct_encoder.top_k,
        num_rbf=configs.model.struct_encoder.num_rbf,
        num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings,
        configs=configs
    )

    visualization_dataset = GVPDataset(
        configs.visualization_settings.data_path,
        seq_mode=configs.model.struct_encoder.use_seq.seq_embed_mode,
        use_rotary_embeddings=configs.model.struct_encoder.use_rotary_embeddings,
        use_foldseek=configs.model.struct_encoder.use_foldseek,
        use_foldseek_vector=configs.model.struct_encoder.use_foldseek_vector,
        top_k=configs.model.struct_encoder.top_k,
        num_rbf=configs.model.struct_encoder.num_rbf,
        num_positional_embeddings=configs.model.struct_encoder.num_positional_embeddings,
        configs=configs
    )
    # train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
    #                           shuffle=configs.train_settings.shuffle,
    #                           num_workers=configs.train_settings.num_workers,
    #                           multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
    #                           pin_memory=True,
    #                           collate_fn=custom_collate)

    train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size, num_workers=8,
                              pin_memory=False,
                              collate_fn=custom_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size, num_workers=2,
                              pin_memory=False,
                              shuffle = False,
                              collate_fn=custom_collate)
    visualization_loader = DataLoader(visualization_dataset, batch_size=configs.visualization_settings.batch_size,
                                      num_workers=1,
                                      pin_memory=False,
                                      shuffle = False,
                                      collate_fn=custom_collate)
    return train_loader, valid_loader, visualization_loader


def prepare_vqvae_dataloaders(logging, accelerator, configs):
    if accelerator.is_main_process:
        logging.info(f"train directory: {configs.train_settings.data_path}")
        logging.info(f"valid directory: {configs.valid_settings.data_path}")
        logging.info(f"visualization directory: {configs.visualization_settings.data_path}")

    train_dataset = VQVAEDataset(configs.train_settings.data_path, train_mode=True, rotate_randomly=False,
                                 configs=configs)
    valid_dataset = VQVAEDataset(configs.valid_settings.data_path, rotate_randomly=False, configs=configs)
    visualization_dataset = VQVAEDataset(configs.visualization_settings.data_path, rotate_randomly=False,
                                         configs=configs)

    train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                              shuffle=configs.train_settings.shuffle,
                              num_workers=configs.train_settings.num_workers,
                              # multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
                              pin_memory=False)

    valid_loader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size,
                              shuffle=False,
                              num_workers=configs.valid_settings.num_workers,
                              # multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
                              pin_memory=False)

    visualization_loader = DataLoader(visualization_dataset, batch_size=configs.visualization_settings.batch_size,
                                      shuffle=False,
                                      num_workers=configs.visualization_settings.num_workers,
                                      pin_memory=True)

    return train_loader, valid_loader, visualization_loader


def prepare_distance_map_vqvae_dataloaders(logging, accelerator, configs):
    if accelerator.is_main_process:
        logging.info(f"train directory: {configs.train_settings.data_path}")
        logging.info(f"valid directory: {configs.valid_settings.data_path}")
        logging.info(f"visualization directory: {configs.visualization_settings.data_path}")

    train_dataset = DistanceMapVQVAEDataset(configs.train_settings.data_path, train_mode=True, rotate_randomly=False,
                                            configs=configs)
    valid_dataset = DistanceMapVQVAEDataset(configs.valid_settings.data_path, rotate_randomly=False, configs=configs)
    visualization_dataset = DistanceMapVQVAEDataset(configs.visualization_settings.data_path, rotate_randomly=False,
                                                    configs=configs)

    train_loader = DataLoader(train_dataset, batch_size=configs.train_settings.batch_size,
                              shuffle=configs.train_settings.shuffle,
                              num_workers=configs.train_settings.num_workers,
                              # multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
                              pin_memory=False)

    valid_loader = DataLoader(valid_dataset, batch_size=configs.valid_settings.batch_size,
                              shuffle=False,
                              num_workers=configs.valid_settings.num_workers,
                              # multiprocessing_context='spawn' if configs.train_settings.num_workers > 0 else None,
                              pin_memory=False)

    visualization_loader = DataLoader(visualization_dataset, batch_size=configs.visualization_settings.batch_size,
                                      shuffle=False,
                                      num_workers=configs.visualization_settings.num_workers,
                                      pin_memory=True)

    return train_loader, valid_loader, visualization_loader


def plot_3d_coords(coords: np.ndarray):
    """
    Plot 3D coordinates in a scatter plot.

    Parameters:
    coords (np.ndarray): A numpy array of shape (N, 3) where N is the number of points,
                         and each point has three coordinates (x, y, z).
    """

    if coords.shape[1] != 3:
        raise ValueError("Input array must have shape (N, 3)")

    # Extracting x, y, and z coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Create a new figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the points
    ax.scatter(x, y, z, c='r', marker='o')

    # Optionally, label the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Show the plot
    plt.show()


def plot_3d_coords_plotly(coords: np.ndarray):
    """
    Plot 3D coordinates in a scatter plot.

    Parameters:
    coords (np.ndarray): A numpy array of shape (N, 3) where N is the number of points,
                         and each point has three coordinates (x, y, z).
    """

    if coords.shape[1] != 3:
        raise ValueError("Input array must have shape (N, 3)")

    # Extracting x, y, and z coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=5, color='red')
    )])

    # Optionally, set titles and labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X axis',
            yaxis_title='Y axis',
            zaxis_title='Z axis'
        ),
        title="3D Scatter Plot"
    )

    # Show the plot
    fig.show()


def plot_3d_coords_lines_plotly(coords: np.ndarray):
    """
    Plot 3D coordinates in a scatter plot and connect each pair of points with a line.

    Parameters:
    coords (np.ndarray): A numpy array of shape (N, 3) where N is the number of points,
                         and each point has three coordinates (x, y, z).
    """
    import plotly.graph_objects as go

    if coords.shape[1] != 3:
        raise ValueError("Input array must have shape (N, 3)")

    # Extracting x, y, and z coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines+markers',  # This will create lines between points
        marker=dict(size=5, color='red')
    )])

    # Optionally, set titles and labels
    fig.update_layout(
        title="3D Scatter Plot with Lines",
        scene=dict(
            xaxis_title="X axis",
            yaxis_title="Y axis",
            zaxis_title="Z axis"
        )
    )

    # Show the plot
    fig.show()


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils.utils import load_configs, get_dummy_logger
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from utils.metrics import batch_distance_map_to_coordinates

    config_path = "../configs/config_distance_map_vqvae.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    # dataset = GVPDataset(test_configs.train_settings.data_path,
    #                      seq_mode=test_configs.model.struct_encoder.use_seq.seq_embed_mode,
    #                      use_rotary_embeddings=test_configs.model.struct_encoder.use_rotary_embeddings,
    #                      use_foldseek=test_configs.model.struct_encoder.use_foldseek,
    #                      use_foldseek_vector=test_configs.model.struct_encoder.use_foldseek_vector,
    #                      top_k=test_configs.model.struct_encoder.top_k,
    #                      num_rbf=test_configs.model.struct_encoder.num_rbf,
    #                      num_positional_embeddings=test_configs.model.struct_encoder.num_positional_embeddings,
    #                      configs=test_configs)

    dataset = DistanceMapVQVAEDataset(test_configs.valid_settings.data_path, train_mode=True, rotate_randomly=False,
                                      configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=test_configs.valid_settings.batch_size, num_workers=0, pin_memory=True,
                             collate_fn=None)

    # test_loader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True)
    struct_embeddings = []
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        # graph = batch["graph"]
        labels = batch['target_distance_map']
        labels = batch_distance_map_to_coordinates(labels.squeeze(1))
        # batch['coords'] = dataset.processor.denormalize_coords(batch['input_coords'][7, ...].squeeze(0).cpu().reshape(-1, 4, 3))
        # plot_3d_coords_lines_plotly(batch["coords"][batch["masks"][7, ...]].cpu().numpy().reshape(-1, 3))
        # plot_3d_coords(batch["coords"][batch["masks"].squeeze(0)].cpu().numpy().reshape(-1, 3))
        # plot_3d_coords_plotly(batch["coords"][batch["masks"]].cpu().numpy().reshape(-1, 3))
        s = DistanceMapVQVAEDataset.create_distance_map(batch["target_coords"][5, ...]).unsqueeze(0)
        plot_3d_coords_lines_plotly(labels[5, ...].cpu().numpy().reshape(-1, 3))
        plot_3d_coords_lines_plotly(batch["target_coords"][5, ...].cpu().numpy().reshape(-1, 3))
        plot_3d_coords_lines_plotly(batch_distance_map_to_coordinates(s).squeeze(0).numpy())
        break
        # pass
