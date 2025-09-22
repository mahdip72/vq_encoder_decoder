"""Utility helpers required by the slimmed GCPNet encoder."""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import Bool, Float, Int64
from torch_geometric.data import Batch
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from ..types import ActivationType
from models.gcpnet.typecheck import jaxtyped, typechecker


def _extract_batch_info(batch_like: Union[Batch, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if isinstance(batch_like, Batch):
        index = batch_like.batch
        num_graphs = batch_like.num_graphs
        lengths = torch.bincount(index, minlength=num_graphs)
    else:
        index = batch_like
        lengths = torch.bincount(index) if index.numel() else index.new_zeros(0, dtype=torch.long)
        num_graphs = lengths.size(0)
    if lengths.device != index.device:
        lengths = lengths.to(index.device)
    return index, lengths, num_graphs


def get_aggregation(aggregation: str) -> Callable:
    def pool_sum(x: torch.Tensor, batch_like: Union[Batch, torch.Tensor]) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, x.size(-1)))
        index, _, num_graphs = _extract_batch_info(batch_like)
        if num_graphs == 0:
            return x.new_zeros((0, x.size(-1)))
        return torch_scatter.scatter(x, index, dim=0, dim_size=num_graphs, reduce="sum")

    def pool_mean(x: torch.Tensor, batch_like: Union[Batch, torch.Tensor]) -> torch.Tensor:
        sums = pool_sum(x, batch_like)
        if sums.size(0) == 0:
            return sums
        _, lengths, _ = _extract_batch_info(batch_like)
        counts = lengths.to(sums.dtype).clamp_min(1).unsqueeze(-1)
        return sums / counts

    def pool_max(x: torch.Tensor, batch_like: Union[Batch, torch.Tensor]) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_zeros((0, x.size(-1)))
        index, _, num_graphs = _extract_batch_info(batch_like)
        if num_graphs == 0:
            return x.new_zeros((0, x.size(-1)))
        return torch_scatter.scatter(x, index, dim=0, dim_size=num_graphs, reduce="max")

    if aggregation == "max":
        return pool_max
    if aggregation == "mean":
        return pool_mean
    if aggregation in {"sum", "add"}:
        return pool_sum
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
    lengths = torch.bincount(batch_index)
    dim_size = lengths.size(0)
    if node_mask is not None:
        centroid = torch_scatter.scatter(
            batch[key][node_mask], batch_index[node_mask], dim=0, reduce="mean", dim_size=dim_size
        )
        centered = torch.full_like(batch[key], torch.inf)
        centered[node_mask] = batch[key][node_mask] - centroid[batch_index][node_mask]
        return centroid, centered

    centroid = torch_scatter.scatter(
        batch[key], batch_index, dim=0, reduce="mean", dim_size=dim_size
    )
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
            batch[key][node_mask]
            + entities_centroid[batch_index][node_mask]
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
