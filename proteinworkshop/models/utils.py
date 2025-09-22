"""Utility helpers required by the slimmed GCPNet encoder."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch_geometric.data import Batch
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from proteinworkshop.types import ActivationType


def get_aggregation(aggregation: str) -> Callable:
    if aggregation == "max":
        return global_max_pool
    if aggregation == "mean":
        return global_mean_pool
    if aggregation in {"sum", "add"}:
        return global_add_pool
    raise ValueError(f"Unknown aggregation function: {aggregation}")


def get_activations(
    act_name: ActivationType, return_functional: bool = False
) -> Union[nn.Module, Callable]:
    if act_name == "relu":
        return F.relu if return_functional else nn.ReLU()
    if act_name == "elu":
        return F.elu if return_functional else nn.ELU()
    if act_name == "leaky_relu":
        return F.leaky_relu if return_functional else nn.LeakyReLU()
    if act_name == "tanh":
        return F.tanh if return_functional else nn.Tanh()
    if act_name == "sigmoid":
        return F.sigmoid if return_functional else nn.Sigmoid()
    if act_name == "none":
        return nn.Identity()
    if act_name in {"silu", "swish"}:
        return F.silu if return_functional else nn.SiLU()
    raise ValueError(f"Unknown activation function: {act_name}")


def flatten_list(lists: List[List]) -> List:
    return [item for sub in lists for item in sub]


@jaxtyped(typechecker=typechecker)
def centralize(
    batch: Union[Batch, ProteinBatch],
    key: str,
    batch_index: torch.Tensor,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if node_mask is not None:
        centroid = torch_scatter.scatter(
            batch[key][node_mask], batch_index[node_mask], dim=0, reduce="mean"
        )
        centered = torch.full_like(batch[key], torch.inf)
        centered[node_mask] = batch[key][node_mask] - centroid[batch_index][node_mask]
        return centroid, centered

    centroid = torch_scatter.scatter(batch[key], batch_index, dim=0, reduce="mean")
    centered = batch[key] - centroid[batch_index]
    return centroid, centered


@jaxtyped(typechecker=typechecker)
def decentralize(
    batch: Union[Batch, ProteinBatch],
    key: str,
    batch_index: torch.Tensor,
    entities_centroid: torch.Tensor,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> torch.Tensor:
    if node_mask is not None:
        restored = torch.full_like(batch[key], torch.inf)
        restored[node_mask] = (
            batch[key][node_mask] + entities_centroid[batch_index][node_mask]
        )
        return restored
    return batch[key] + entities_centroid[batch_index]


@jaxtyped(typechecker=typechecker)
def localize(
    pos: Float[torch.Tensor, "batch_num_nodes 3"],
    edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
    norm_pos_diff: bool = True,
    node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> Float[torch.Tensor, "batch_num_edges 3 3"]:
    row, col = edge_index

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]
        pos_diff = torch.full((edge_index.size(1), 3), torch.inf, device=pos.device)
        pos_cross = torch.full_like(pos_diff, torch.inf)
        pos_diff[edge_mask] = pos[row][edge_mask] - pos[col][edge_mask]
        pos_cross[edge_mask] = torch.cross(pos[row][edge_mask], pos[col][edge_mask])
    else:
        pos_diff = pos[row] - pos[col]
        pos_cross = torch.cross(pos[row], pos[col])

    if norm_pos_diff:
        if node_mask is not None:
            norm = torch.ones((edge_index.size(1), 1), device=pos.device)
            norm[edge_mask] = pos_diff[edge_mask].norm(dim=1, keepdim=True) + 1
        else:
            norm = pos_diff.norm(dim=1, keepdim=True) + 1
        pos_diff = pos_diff / norm

        if node_mask is not None:
            cross_norm = torch.ones((edge_index.size(1), 1), device=pos.device)
            cross_norm[edge_mask] = pos_cross[edge_mask].norm(dim=1, keepdim=True) + 1
        else:
            cross_norm = pos_cross.norm(dim=1, keepdim=True) + 1
        pos_cross = pos_cross / cross_norm

    if node_mask is not None:
        pos_vertical = torch.full_like(pos_diff, torch.inf)
        pos_vertical[edge_mask] = torch.cross(
            pos_diff[edge_mask], pos_cross[edge_mask]
        )
    else:
        pos_vertical = torch.cross(pos_diff, pos_cross)

    return torch.cat(
        (
            pos_diff.unsqueeze(1),
            pos_cross.unsqueeze(1),
            pos_vertical.unsqueeze(1),
        ),
        dim=1,
    )


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
        norm = torch.sqrt(norm.clamp_min(eps))
    return norm



def is_identity(obj: Union[nn.Module, Callable]) -> bool:
    return isinstance(obj, nn.Identity) or getattr(obj, "__name__", None) == "identity"
