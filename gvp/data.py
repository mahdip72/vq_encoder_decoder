import glob
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import tqdm
import os
from torch.utils.data import DataLoader, Dataset
from utils import load_h5_file
from torch_geometric.data import Batch, Data


def custom_collate(one_batch):
    # Unpack the batch
    data_list = [item[0] for item in one_batch]

    # Create a Batch object
    one_batch = Batch.from_data_list(data_list)

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
    d_mu = torch.linspace(d_min, d_max, d_count, device=device)
    d_mu = d_mu.view([1, -1])
    d_sigma = (d_max - d_min) / d_count
    d_sigma = torch.tensor(d_sigma, device=device)  # Convert d_sigma to a tensor
    d_expand = torch.unsqueeze(d, -1)

    RBF = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
    return RBF


class ProteinGraphDataset(Dataset):
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
                 top_k=30, num_rbf=16, device="cuda"):
        super(ProteinGraphDataset, self).__init__()

        self.h5_samples = glob.glob(os.path.join(data_path, '*.h5'))
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        # self.node_counts = [len(e['seq']) for e in data_list]

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}

    def __len__(self):
        return len(self.h5_samples)

    def __getitem__(self, i):
        sample_path = self.h5_samples[i]
        sample = load_h5_file(sample_path)
        sample_dict = {'name': os.path.basename(sample_path).split('-')[1],
                       'coords': sample[1].tolist(),
                       'seq': sample[0].decode('utf-8')}
        feature = self._featurize_as_graph(sample_dict)
        return [feature]

    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'],
                                     device=self.device, dtype=torch.float32)
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                  device=self.device, dtype=torch.long)

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), d_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
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
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

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
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
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
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec


if __name__ == '__main__':

    dataset_path = './save_test'
    dataset = ProteinGraphDataset(dataset_path)

    test_dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=6, collate_fn=custom_collate)

    print("Testing on your dataset")
    for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader)):
        batch = batch.to('cuda')  # Move the batch to the device
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        e_E = batch.edge_index
        # sample = model.sample(h_V, batch.edge_index, h_E, n_samples=100)
        # Continue with the rest of your processing...

    print('done')
