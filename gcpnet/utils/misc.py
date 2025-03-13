import huggingface_hub
import math
import os
import torch
import torch_scatter

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from beartype import beartype as typechecker
from collections import defaultdict
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch_geometric.data import Batch
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from typing import Any, Callable, ContextManager, Optional, Sequence, Tuple, TypeVar, Union
from warnings import warn

from gcpnet.utils.constants.esm3 import CHAIN_BREAK_STR
from gcpnet.utils.types import FunctionAnnotation

MAX_SUPPORTED_DISTANCE = 1e6


TSequence = TypeVar("TSequence", bound=Sequence)


def slice_python_object_as_numpy(
    obj: TSequence, idx: int | list[int] | slice | np.ndarray
) -> TSequence:
    """
    Slice a python object (like a list, string, or tuple) as if it was a numpy object.

    Example:
        >>> obj = "ABCDE"
        >>> slice_python_object_as_numpy(obj, [1, 3, 4])
        "BDE"

        >>> obj = [1, 2, 3, 4, 5]
        >>> slice_python_object_as_numpy(obj, np.arange(5) < 3)
        [1, 2, 3]
    """
    if isinstance(idx, int):
        idx = [idx]

    if isinstance(idx, np.ndarray) and idx.dtype == bool:
        sliced_obj = [obj[i] for i in np.where(idx)[0]]
    elif isinstance(idx, slice):
        sliced_obj = obj[idx]
    else:
        sliced_obj = [obj[i] for i in idx]

    match obj, sliced_obj:
        case str(), list():
            sliced_obj = "".join(sliced_obj)
        case _:
            sliced_obj = obj.__class__(sliced_obj)  # type: ignore

    return sliced_obj  # type: ignore


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(
        v_min, v_max, n_bins, device=values.device, dtype=values.dtype
    )
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-(z**2))


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return batched_gather(s.unsqueeze(-3), edges, -2, no_batch_dims=len(s.shape) - 1)


def knn_graph(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    sequence_id: torch.Tensor,
    *,
    no_knn: int,
):
    L = coords.shape[-2]
    num_by_dist = min(no_knn, L)
    device = coords.device

    coords = coords.nan_to_num()
    coord_mask = ~(coord_mask[..., None, :] & coord_mask[..., :, None])
    padding_pairwise_mask = padding_mask[..., None, :] | padding_mask[..., :, None]
    if sequence_id is not None:
        padding_pairwise_mask |= torch.unsqueeze(sequence_id, 1) != torch.unsqueeze(
            sequence_id, 2
        )
    dists = (coords.unsqueeze(-2) - coords.unsqueeze(-3)).norm(dim=-1)
    arange = torch.arange(L, device=device)
    seq_dists = (arange.unsqueeze(-1) - arange.unsqueeze(-2)).abs()
    # We only support up to a certain distance, above that, we use sequence distance
    # instead. This is so that when a large portion of the structure is masked out,
    # the edges are built according to sequence distance.
    max_dist = MAX_SUPPORTED_DISTANCE
    torch._assert_async((dists[~coord_mask] < max_dist).all())
    struct_then_seq_dist = (
        seq_dists.to(dists.dtype)
        .mul(1e2)
        .add(max_dist)
        .where(coord_mask, dists)
        .masked_fill(padding_pairwise_mask, torch.inf)
    )
    dists, edges = struct_then_seq_dist.sort(dim=-1, descending=False)
    # This is a L x L tensor, where we index by rows first,
    # and columns are the edges we should pick.
    chosen_edges = edges[..., :num_by_dist]
    chosen_mask = dists[..., :num_by_dist].isfinite()
    return chosen_edges, chosen_mask


def stack_variable_length_tensors(
    sequences: Sequence[torch.Tensor],
    constant_value: int | float = 0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Automatically stack tensors together, padding variable lengths with the
    value in constant_value. Handles an arbitrary number of dimensions.

    Examples:
        >>> tensor1, tensor2 = torch.ones([2]), torch.ones([5])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5]. First row is [1, 1, 0, 0, 0]. Second row is all ones.

        >>> tensor1, tensor2 = torch.ones([2, 4]), torch.ones([5, 3])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5, 4]
    """
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype
    device = sequences[0].device

    array = torch.full(shape, constant_value, dtype=dtype, device=device)
    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


def unbinpack(
    tensor: torch.Tensor, sequence_id: torch.Tensor | None, pad_value: int | float
):
    """
    Args:
        tensor (Tensor): [B, L, ...]

    Returns:
        Tensor: [B_unbinpacked, L_unbinpack, ...]
    """
    if sequence_id is None:
        return tensor

    unpacked_tensors = []
    num_sequences = sequence_id.max(dim=-1).values + 1
    for batch_idx, (batch_seqid, batch_num_sequences) in enumerate(
        zip(sequence_id, num_sequences)
    ):
        for seqid in range(batch_num_sequences):
            mask = batch_seqid == seqid
            unpacked = tensor[batch_idx, mask]
            unpacked_tensors.append(unpacked)
    return stack_variable_length_tensors(unpacked_tensors, pad_value)


def fp32_autocast_context(device_type: str) -> ContextManager[torch.amp.autocast]:  # type: ignore
    """
    Returns an autocast context manager that disables downcasting by AMP.

    Args:
        device_type: The device type ('cpu' or 'cuda')

    Returns:
        An autocast context manager with the specified behavior.
    """
    if device_type == "cpu":
        return torch.amp.autocast(device_type, enabled=False)  # type: ignore
    elif device_type == "cuda":
        return torch.amp.autocast(device_type, dtype=torch.float32)  # type: ignore
    else:
        raise ValueError(f"Unsupported device type: {device_type}")


def merge_ranges(ranges: list[range], merge_gap_max: int | None = None) -> list[range]:
    """Merge overlapping ranges into sorted, non-overlapping segments.

    Args:
        ranges: collection of ranges to merge.
        merge_gap_max: optionally merge neighboring ranges that are separated by a gap
          no larger than this size.
    Returns:
        non-overlapping ranges merged from the inputs, sorted by position.
    """
    ranges = sorted(ranges, key=lambda r: r.start)
    merge_gap_max = merge_gap_max if merge_gap_max is not None else 0
    assert merge_gap_max >= 0, f"Invalid merge_gap_max: {merge_gap_max}"

    merged = []
    for r in ranges:
        if not merged:
            merged.append(r)
        else:
            last = merged[-1]
            if last.stop + merge_gap_max >= r.start:
                merged[-1] = range(last.start, max(last.stop, r.stop))
            else:
                merged.append(r)
    return merged


def merge_annotations(
    annotations: list[FunctionAnnotation],
    merge_gap_max: int | None = None,
) -> list[FunctionAnnotation]:
    """Merges annotations into non-overlapping segments.

    Args:
        annotations: annotations to merge.
        merge_gap_max: optionally merge neighboring ranges that are separated by a gap
          no larger than this size.
    Returns:
        non-overlapping annotations with gaps merged.
    """
    grouped: dict[str, list[range]] = defaultdict(list)
    for a in annotations:
        # +1 since FunctionAnnotation.end is inlcusive.
        grouped[a.label].append(range(a.start, a.end + 1))

    merged = []
    for label, ranges in grouped.items():
        merged_ranges = merge_ranges(ranges, merge_gap_max=merge_gap_max)
        for range_ in merged_ranges:
            annotation = FunctionAnnotation(
                label=label,
                start=range_.start,
                end=range_.stop - 1,  # convert range.stop exclusive -> inclusive.
            )
            merged.append(annotation)
    return merged


def list_nan_to_none(l: list) -> list:
    if l is None:
        return None  # type: ignore
    elif isinstance(l, float):
        return None if math.isnan(l) else l  # type: ignore
    elif isinstance(l, list):
        return [list_nan_to_none(x) for x in l]
    else:
        # Don't go into other structures.
        return l


def list_none_to_nan(l: list) -> list:
    if l is None:
        return math.nan  # type: ignore
    elif isinstance(l, list):
        return [list_none_to_nan(x) for x in l]
    else:
        return l


def maybe_tensor(x, convert_none_to_nan: bool = False) -> torch.Tensor | None:
    if x is None:
        return None
    if convert_none_to_nan:
        x = list_none_to_nan(x)
    return torch.tensor(x)


def maybe_list(x, convert_nan_to_none: bool = False) -> list | None:
    if x is None:
        return None
    x = x.tolist()
    if convert_nan_to_none:
        x = list_nan_to_none(x)
    return x


def huggingfacehub_login():
    """Authenticates with the Hugging Face Hub using the HF_TOKEN environment
    variable, else by prompting the user"""
    token = os.environ.get("HF_TOKEN")
    huggingface_hub.login(token=token)


def get_chainbreak_boundaries_from_sequence(sequence: Sequence[str]) -> np.ndarray:
    chain_boundaries = [0]
    for i, aa in enumerate(sequence):
        if aa == CHAIN_BREAK_STR:
            if i == (len(sequence) - 1):
                raise ValueError(
                    "Encountered chain break token at end of sequence, this is unexpected."
                )
            if i == (len(sequence) - 2):
                warn(
                    "Encountered chain break token at penultimate position, this is unexpected."
                )
            chain_boundaries.append(i)
            chain_boundaries.append(i + 1)
    chain_boundaries.append(len(sequence))
    assert len(chain_boundaries) % 2 == 0
    chain_boundaries = np.array(chain_boundaries).reshape(-1, 2)
    return chain_boundaries


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
