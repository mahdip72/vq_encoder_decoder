from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from beartype import beartype as typechecker
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch_geometric.data import Batch
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


def get_aggregation(aggregation: str) -> Callable:
    """Maps aggregation name (str) to aggregation function."""
    if aggregation == "max":
        return global_max_pool
    elif aggregation == "mean":
        return global_mean_pool
    elif aggregation in {"sum", "add"}:
        return global_add_pool
    else:
        raise ValueError(f"Unknown aggregation function: {aggregation}")


def get_activations(
    act_name: Any, return_functional: bool = False
) -> Union[nn.Module, Callable]:
    """Maps activation name (str) to activation function module."""
    if act_name == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif act_name == "elu":
        return F.elu if return_functional else nn.ELU()
    elif act_name == "leaky_relu":
        return F.leaky_relu if return_functional else nn.LeakyReLU()
    elif act_name == "tanh":
        return F.tanh if return_functional else nn.Tanh()
    elif act_name == "sigmoid":
        return F.sigmoid if return_functional else nn.Sigmoid()
    elif act_name == "none":
        return nn.Identity()
    elif act_name in {"silu", "swish"}:
        return F.silu if return_functional else nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {act_name}")


@jaxtyped(typechecker=typechecker)
def centralize(
    batch: Batch,
    key: str,
    batch_index: torch.Tensor,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor
]:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        # derive centroid of each batch element
        entities_centroid = torch_scatter.scatter(
            batch[key][node_mask], batch_index[node_mask], dim=0, reduce="mean"
        )  # e.g., [batch_size, 3]

        # center entities using corresponding centroids
        entities_centered = batch[key] - (
            entities_centroid[batch_index] * node_mask.float().unsqueeze(-1)
        )
        masked_values = torch.ones_like(batch[key]) * torch.inf
        values = batch[key][node_mask]
        masked_values[node_mask] = (
            values - entities_centroid[batch_index][node_mask]
        )
        entities_centered = masked_values

    else:
        # derive centroid of each batch element, and center entities using corresponding centroids
        entities_centroid = torch_scatter.scatter(
            batch[key], batch_index, dim=0, reduce="mean"
        )  # e.g., [batch_size, 3]
        entities_centered = batch[key] - entities_centroid[batch_index]

    return entities_centroid, entities_centered


@jaxtyped(typechecker=typechecker)
def decentralize(
    batch: Batch,
    key: str,
    batch_index: torch.Tensor,
    entities_centroid: torch.Tensor,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> torch.Tensor:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        masked_values = torch.ones_like(batch[key]) * torch.inf
        masked_values[node_mask] = (
            batch[key][node_mask] + entities_centroid[batch_index]
        )
        entities_centered = masked_values
    else:
        entities_centroid = entities_centroid.unsqueeze(-2) if batch[key].ndim == 3 else entities_centroid
        entities_centered = batch[key] + entities_centroid[batch_index]
    return entities_centered


@jaxtyped(typechecker=typechecker)
def localize(
    pos: Float[torch.Tensor, "batch_num_nodes 3"],
    edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
    norm_pos_diff: bool = True,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> Float[torch.Tensor, "batch_num_edges 3 3"]:
    row, col = edge_index[0], edge_index[1]

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

        pos_diff = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        pos_diff[edge_mask] = pos[row][edge_mask] - pos[col][edge_mask]

        pos_cross = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        pos_cross[edge_mask] = torch.cross(
            pos[row][edge_mask], pos[col][edge_mask], dim=-1
        )
    else:
        pos_diff = pos[row] - pos[col]
        pos_cross = torch.cross(pos[row], pos[col], dim=-1)

    if norm_pos_diff:
        # derive and apply normalization factor for `pos_diff`
        if node_mask is not None:
            norm = torch.ones((edge_index.shape[1], 1), device=pos_diff.device)
            norm[edge_mask] = (
                torch.sqrt(
                    torch.sum((pos_diff[edge_mask] ** 2), dim=1).unsqueeze(1)
                )
            ) + 1
        else:
            norm = torch.sqrt(torch.sum(pos_diff**2, dim=1).unsqueeze(1)) + 1
        pos_diff = pos_diff / norm

        # derive and apply normalization factor for `pos_cross`
        if node_mask is not None:
            cross_norm = torch.ones(
                (edge_index.shape[1], 1), device=pos_cross.device
            )
            cross_norm[edge_mask] = (
                torch.sqrt(
                    torch.sum((pos_cross[edge_mask]) ** 2, dim=1).unsqueeze(1)
                )
            ) + 1
        else:
            cross_norm = (
                torch.sqrt(torch.sum(pos_cross**2, dim=1).unsqueeze(1))
            ) + 1
        pos_cross = pos_cross / cross_norm

    if node_mask is not None:
        pos_vertical = (
            torch.ones((edge_index.shape[1], 3), device=edge_index.device)
            * torch.inf
        )
        pos_vertical[edge_mask] = torch.cross(
            pos_diff[edge_mask], pos_cross[edge_mask], dim=-1
        )
    else:
        pos_vertical = torch.cross(pos_diff, pos_cross, dim=-1)

    f_ij = torch.cat(
        (
            pos_diff.unsqueeze(1),
            pos_cross.unsqueeze(1),
            pos_vertical.unsqueeze(1),
        ),
        dim=1,
    )
    return f_ij


@jaxtyped(typechecker=typechecker)
def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True,
) -> torch.Tensor:
    norm = torch.sum(x**2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm + eps)
    return norm + eps


@jaxtyped(typechecker=typechecker)
def is_identity(
    nonlinearity: Optional[Union[Callable, nn.Module]] = None
) -> bool:
    return nonlinearity is None or isinstance(nonlinearity, nn.Identity)


@jaxtyped(typechecker=typechecker)
def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Safely normalize a Tensor. Adapted from:
    https://github.com/drorlab/gvp-pytorch.

    :param tensor: Tensor of any shape.
    :type tensor: Tensor
    :param dim: The dimension over which to normalize the input Tensor.
    :type dim: int, optional
    :return: The normalized Tensor.
    :rtype: torch.Tensor
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


@jaxtyped(typechecker=typechecker)
def batch_orientations(
    X: torch.Tensor, coords_slice_index: torch.Tensor, ca_idx: int = 1
) -> torch.Tensor:
    """
    Compute forward and backward orientation vectors for each node
    in batched graph of protein structures.

    From:
    https://github.com/a-r-j/ProteinWorkshop/blob/main/proteinworkshop/features/node_features.py
    
    :param X: The input node features.
    :type X: torch.Tensor
    :param coords_slice_index: The slice index for the coordinates.
    :type coords_slice_index: torch.Tensor
    :param ca_idx: The index of the alpha carbon atom.
    :type ca_idx: int, optional
    :return: The forward and backward orientation vectors.
    :rtype: torch.Tensor
    """
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
