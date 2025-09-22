"""Node feature computation functions."""
from typing import List, Union

import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from graphein.protein.tensor.angles import alpha, dihedrals, kappa
from graphein.protein.tensor.data import Protein, ProteinBatch
from graphein.protein.tensor.types import AtomTensor, CoordTensor
from jaxtyping import jaxtyped
try:  # Optional dependency for Hydra-style configs
    from omegaconf import ListConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when OmegaConf is unavailable
    ListConfig = list  # type: ignore
from torch_geometric.data import Batch, Data

from ..types import OrientationTensor, ScalarNodeFeature

from .sequence_features import amino_acid_one_hot
from .utils import _normalize


@jaxtyped(typechecker=typechecker)
def compute_scalar_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    node_features: Union[ListConfig, List[ScalarNodeFeature]],
) -> torch.Tensor:
    """
    Factory function for node features.

    .. seealso::
        :py:class:`models.gcpnet.types.ScalarNodeFeature` for a list of node
        features that can be computed.

    This function operates on a :py:class:`torch_geometric.data.Data` or
    :py:class:`torch_geometric.data.Batch` object and computes the requested
    node features.

    :param x: :py:class:`~torch_geometric.data.Data` or
        :py:class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param node_features: List of node features to compute.
    :type node_features: Union[List[str], ListConfig]
    :return: Tensor of node features of shape (``N x F``), where ``N`` is the
        number of nodes and ``F`` is the number of features.
    :rtype: torch.Tensor
    """
    feats = []
    for feature in node_features:
        if feature == "amino_acid_one_hot":
            feats.append(amino_acid_one_hot(x, num_classes=23))
        elif feature == "alpha":
            feats.append(alpha(x.coords, x.batch, rad=True, embed=True))
        elif feature == "kappa":
            feats.append(kappa(x.coords, x.batch, rad=True, embed=True))
        elif feature == "dihedrals":
            feats.append(dihedrals(x.coords, x.batch, rad=True, embed=True))
        elif feature == "sequence_positional_encoding":
            continue
        else:
            raise ValueError(f"Node feature {feature} not recognised.")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    # Return concatenated features or original features if no features were computed
    return torch.cat(feats, dim=1) if feats else x.x


@jaxtyped(typechecker=typechecker)
def compute_vector_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    vector_features: Union[ListConfig, List[str]],
) -> Union[Batch, Data, Protein, ProteinBatch]:
    """Factory function for vector features.

    Currently implemented vector features are:

        - ``orientation``: Orientation of each node in the protein backbone
        - ``virtual_cb_vector``: Virtual CB vector for each node in the protein
        backbone


    """
    vector_node_features = []
    for feature in vector_features:
        if feature == "orientation":
            vector_node_features.append(orientations(x.coords, x._slice_dict["coords"]))
        elif feature == "virtual_cb_vector":
            raise NotImplementedError("Virtual CB vector not implemented yet.")
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    x.x_vector_attr = torch.cat(vector_node_features, dim=0)
    return x


@jaxtyped(typechecker=typechecker)
def orientations(
    X: Union[CoordTensor, AtomTensor], coords_slice_index: torch.Tensor, ca_idx: int = 1
) -> OrientationTensor:
    if X.ndim == 3:
        X = X[:, ca_idx, :]

    # NOTE: the first item in the coordinates slice index is always 0,
    # and the last item is always the node count of the batch
    batch_num_nodes = X.shape[0]
    slice_index = coords_slice_index[1:] - 1
    last_node_index = slice_index[:-1]
    first_node_index = slice_index[:-1] + 1
    slice_mask = torch.zeros(batch_num_nodes - 1, dtype=torch.bool)
    last_node_forward_slice_mask = slice_mask.clone()
    first_node_backward_slice_mask = slice_mask.clone()

    # NOTE: all of the last (first) nodes in a subgraph have their
    # forward (backward) vectors set to a padding value (i.e., 0.0)
    # to mimic feature construction behavior with single input graphs
    forward_slice = X[1:] - X[:-1]
    backward_slice = X[:-1] - X[1:]
    last_node_forward_slice_mask[last_node_index] = True
    first_node_backward_slice_mask[first_node_index - 1] = True  # NOTE: for the backward slices, our indexing defaults to node index `1`
    forward_slice[last_node_forward_slice_mask] = 0.0 # NOTE: this handles all but the last node in the last subgraph
    backward_slice[first_node_backward_slice_mask] = 0.0 # NOTE: this handles all but the first node in the first subgraph

    # NOTE: padding first and last nodes with zero vectors does not impact feature normalization
    forward = _normalize(forward_slice)
    backward = _normalize(backward_slice)
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    orientations = torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)

    # optionally debug/verify the orientations
    # last_node_indices = torch.cat((last_node_index, torch.tensor([batch_num_nodes - 1])), dim=0)
    # first_node_indices = torch.cat((torch.tensor([0]), first_node_index), dim=0)
    # intermediate_node_indices_mask = torch.ones(batch_num_nodes, device=X.device, dtype=torch.bool)
    # intermediate_node_indices_mask[last_node_indices] = False
    # intermediate_node_indices_mask[first_node_indices] = False
    # assert not orientations[last_node_indices][:, 0].any() and orientations[last_node_indices][:, 1].any()
    # assert orientations[first_node_indices][:, 0].any() and not orientations[first_node_indices][:, 1].any()
    # assert orientations[intermediate_node_indices_mask][:, 0].any() and orientations[intermediate_node_indices_mask][:, 1].any()

    return orientations
