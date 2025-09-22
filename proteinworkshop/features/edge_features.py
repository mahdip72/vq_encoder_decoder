"""Minimal edge feature computation helpers."""

from typing import List, Union

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from jaxtyping import jaxtyped
try:  # Optional dependency for Hydra-style configs
    from omegaconf import ListConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when OmegaConf is unavailable
    ListConfig = list  # type: ignore
from torch_geometric.data import Batch, Data

from proteinworkshop.features.utils import _normalize


@jaxtyped(typechecker=typechecker)
def compute_scalar_edge_features(
    x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> torch.Tensor:
    """Return scalar edge features; only ``edge_distance`` is supported."""

    scalars = []
    for feature in features:
        if feature == "edge_distance":
            scalars.append(_edge_distance(x.pos, x.edge_index))
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported scalar edge feature: {feature}")

    return torch.cat(scalars, dim=1) if scalars else torch.empty(0)


@jaxtyped(typechecker=typechecker)
def compute_vector_edge_features(
    x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> Union[Data, Batch]:
    """Return vector edge features; only ``edge_vectors`` is supported."""

    vectors = []
    for feature in features:
        if feature != "edge_vectors":  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported vector edge feature: {feature}")
        diff = x.pos[x.edge_index[0]] - x.pos[x.edge_index[1]]
        vectors.append(_normalize(diff).unsqueeze(-2))

    x.edge_vector_attr = torch.cat(vectors, dim=0)
    return x


@jaxtyped(typechecker=typechecker)
def _edge_distance(
    pos: CoordTensor,
    edge_index: EdgeTensor,
) -> torch.Tensor:
    return torch.pairwise_distance(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(-1)
