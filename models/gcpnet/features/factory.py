from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn
from graphein.protein.tensor.data import ProteinBatch, get_random_batch
from torch_geometric.data import Batch
from torch_geometric.nn import knn_graph
import torch._dynamo as dynamo
from torch_geometric.nn.encoding import PositionalEncoding

from .edge_features import (
    compute_scalar_edge_features,
    compute_vector_edge_features,
)
from .node_features import (
    compute_scalar_node_features,
    compute_vector_node_features,
)
from .representation import transform_representation
from ..typecheck import jaxtyped, typechecker
from ..types import (
    ScalarEdgeFeature,
    ScalarNodeFeature,
    VectorEdgeFeature,
    VectorNodeFeature,
)

StructureRepresentation = Literal["ca", "ca_bb", "full_atom"]


class ProteinFeaturiser(nn.Module):
    """
    Initialise a protein featuriser.

    :param representation: Representation to use for the protein.
        One of ``"ca", "ca_bb", "full_atom"``.
    :type representation: StructureRepresentation
    :param scalar_node_features: List of scalar-values node features to
        compute. Options: ``"amino_acid_one_hot",
        "sequence_positional_encoding", "alpha", "kappa", "dihedrals"
        "sidechain_torsions"``.
    :type scalar_node_features: List[ScalarNodeFeature]
    :param vector_node_features: List of vector-valued node features to
        compute. # TODO types
    :type vector_node_features: List[VectorNodeFeature]
    :param edge_types: List of edge types to compute.
        Options: # TODO types
    :type edge_types: List[str]
    :param scalar_edge_features: List of scalar-valued edge features to
        compute. # TODO types
    :type scalar_edge_features: List[ScalarEdgeFeature]
    :param vector_edge_features: List of vector-valued edge features to
        compute. # TODO types
    :type vector_edge_features: List[VectorEdgeFeature]
    """

    def __init__(
        self,
        representation: StructureRepresentation,
        scalar_node_features: List[ScalarNodeFeature],
        vector_node_features: List[VectorNodeFeature],
        edge_types: List[str],
        scalar_edge_features: List[ScalarEdgeFeature],
        vector_edge_features: List[VectorEdgeFeature],
    ):
        super(ProteinFeaturiser, self).__init__()
        self.representation = representation
        self.scalar_node_features = scalar_node_features
        self.vector_node_features = vector_node_features
        self.edge_types = edge_types
        self.scalar_edge_features = scalar_edge_features
        self.vector_edge_features = vector_edge_features

        if "sequence_positional_encoding" in self.scalar_node_features:
            self.positional_encoding = PositionalEncoding(16)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Union[Batch, ProteinBatch]:
        # Scalar node features
        if self.scalar_node_features:
            concat_nf = False
            if hasattr(self, "positional_encoding"):
                batch.x = self.positional_encoding(batch.seq_pos)
                # This is necessary to concat node features with the positional encoding
                concat_nf = True
            if self.scalar_node_features != ["sequence_positional_encoding"]:
                scalar_features = compute_scalar_node_features(
                    batch, self.scalar_node_features
                )
                if concat_nf:
                    batch.x = torch.cat([batch.x, scalar_features], dim=-1)
                else:
                    batch.x = scalar_features
            batch.x = torch.nan_to_num(
                batch.x, nan=0.0, posinf=0.0, neginf=0.0
            )

        # Representation
        batch = transform_representation(batch, self.representation)

        # Vector node features
        if self.vector_node_features:
            batch = compute_vector_node_features(
                batch, self.vector_node_features
            )

        # Edges
        if self.edge_types:
            batch.edge_index, batch.edge_type = _compute_edges(
                batch, self.edge_types
            )
            batch.num_relation = len(self.edge_types)

        # Scalar edge features
        if self.scalar_edge_features:
            batch.edge_attr = compute_scalar_edge_features(
                batch, self.scalar_edge_features
            )

        # Vector edge features
        if self.vector_edge_features:
            batch = compute_vector_edge_features(
                batch, self.vector_edge_features
            )

        return batch

    def _example(self, batch_size: int = 2):
        batch = get_random_batch(batch_size)
        return self(batch)

    def __repr__(self) -> str:
        return f"ProteinFeaturiser(representation={self.representation}, scalar_node_features={self.scalar_node_features}, vector_node_features={self.vector_node_features}, edge_types={self.edge_types}, scalar_edge_features={self.scalar_edge_features}, vector_edge_features={self.vector_edge_features})"

def _compute_edges(
    batch: Batch, edge_types: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minimal edge constructor supporting knn graphs for CA backbones."""

    if len(edge_types) != 1 or not edge_types[0].startswith("knn_"):
        raise ValueError(
            "Trimmed ProteinWorkshop supports only a single 'knn_{k}' edge type"
        )

    try:
        k = int(edge_types[0].split("_", 1)[1])
    except (IndexError, ValueError) as exc:  # pragma: no cover - defensive parsing
        raise ValueError(
            f"Invalid edge type specification: {edge_types[0]}"
        ) from exc

    if getattr(batch, 'edge_index_precomputed', False):
        edge_index = batch.edge_index
        edge_type = getattr(batch, 'edge_type', None)
        if edge_type is None or edge_type.numel() == 0:
            edge_type = torch.zeros(
                edge_index.size(1), dtype=torch.long, device=edge_index.device
            )
        return edge_index, edge_type

    edge_index = _knn_graph_eager(
        x=batch.pos,
        k=k,
        batch=batch.batch,
        loop=False,
    )
    edge_type = torch.zeros(
        edge_index.size(1), dtype=torch.long, device=edge_index.device
    )
    return edge_index, edge_type

@dynamo.disable
def _knn_graph_eager(*args, **kwargs):
    """Wrapper to keep torch_cluster-based knn_graph out of torch.compile traces."""
    return knn_graph(*args, **kwargs)
