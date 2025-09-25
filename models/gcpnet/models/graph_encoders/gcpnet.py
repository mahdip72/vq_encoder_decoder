from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn as nn
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch

from .layers import gcp
from .components.wrappers import (
    ScalarVector,
)
from ..utils import (
    centralize,
    decentralize,
    get_aggregation,
    localize,
)
from ...types import EncoderOutput
from models.gcpnet.typecheck import jaxtyped, typechecker


class GCPNetModel(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 5,
        node_s_emb_dim: int = 128,
        node_v_emb_dim: int = 16,
        edge_s_emb_dim: int = 32,
        edge_v_emb_dim: int = 4,
        r_max: float = 10.0,
        num_rbf: int = 8,
        activation: str = "silu",
        pool: str = "sum",
        # Note: Each of the arguments above are stored in the corresponding `kwargs` configs below
        # They are simply listed here to highlight key available arguments
        **kwargs,
    ):
        """
        Initializes an instance of the GCPNetModel class with the provided
        parameters.
        Note: Each of the model's keyword arguments listed here
        are also referenced in the corresponding `DictConfigs` within `kwargs`.
        They are simply listed here to highlight some of the key arguments available.
        See `models/gcpnet/config/encoder/gcpnet.yaml` for a full list of all available arguments.

        :param num_layers: Number of layers in the model (default: ``5``)
        :type num_layers: int
        :param node_s_emb_dim: Dimension of the node state embeddings (default: ``128``)
        :type node_s_emb_dim: int
        :param node_v_emb_dim: Dimension of the node vector embeddings (default: ``16``)
        :type node_v_emb_dim: int
        :param edge_s_emb_dim: Dimension of the edge state embeddings
            (default: ``32``)
        :type edge_s_emb_dim: int
        :param edge_v_emb_dim: Dimension of the edge vector embeddings
            (default: ``4``)
        :type edge_v_emb_dim: int
        :param r_max: Maximum distance for radial basis functions
            (default: ``10.0``)
        :type r_max: float
        :param num_rbf: Number of radial basis functions (default: ``8``)
        :type num_rbf: int
        :param activation: Activation function to use in each GCP layer (default: ``silu``)
        :type activation: str
        :param pool: Global pooling method to be used
            (default: ``"sum"``)
        :type pool: str
        :param kwargs: Primary model arguments in the form of the
            `DictConfig`s `module_cfg`, `model_cfg`, and `layer_cfg`, respectively
        :type kwargs: dict
        """
        super().__init__()

        assert all(
            [cfg in kwargs for cfg in ["module_cfg", "model_cfg", "layer_cfg"]]
        ), "All required GCPNet `DictConfig`s must be provided."
        module_cfg = kwargs["module_cfg"]
        model_cfg = kwargs["model_cfg"]
        layer_cfg = kwargs["layer_cfg"]

        self.predict_node_pos = module_cfg.predict_node_positions
        self.predict_node_rep = module_cfg.predict_node_rep

        # Feature dimensionalities
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = ScalarVector(model_cfg.h_input_dim, model_cfg.chi_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)

        # Position-wise operations
        self.centralize = partial(centralize, key="pos")
        self.localize = partial(localize, norm_pos_diff=module_cfg.norm_pos_diff)
        self.decentralize = partial(decentralize, key="pos")
        self._frame_update_eps = 1e-6

        # Input embeddings
        self.gcp_embedding = gcp.GCPEmbedding(
            edge_input_dims,
            node_input_dims,
            self.edge_dims,
            self.node_dims,
            cfg=module_cfg,
        )

        # Message-passing layers
        self.interaction_layers = nn.ModuleList(
            gcp.GCPInteractions(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.num_layers)
        )

        if self.predict_node_rep:
            # Predictions
            self.invariant_node_projection = nn.ModuleList(
                [
                    gcp.GCPLayerNorm(self.node_dims),
                    gcp.GCP(
                        # Note: `GCPNet` defaults to providing SE(3) equivariance
                        # It is possible to provide E(3) equivariance by instead setting `module_cfg.enable_e3_equivariance=true`
                        self.node_dims,
                        (self.node_dims.scalar, 0),
                        nonlinearities=tuple(module_cfg.nonlinearities),
                        scalar_gate=module_cfg.scalar_gate,
                        vector_gate=module_cfg.vector_gate,
                        enable_e3_equivariance=module_cfg.enable_e3_equivariance,
                        node_inputs=True,
                    ),
                ]
            )

        # Global pooling/readout function
        self.readout = get_aggregation(
            module_cfg.pool
        )  # {"mean": global_mean_pool, "sum": global_add_pool}[pool]

    @property
    def required_batch_attributes(self) -> List[str]:
        return ["edge_index", "pos", "x", "batch"]

    def _ensure_edge_frames(
        self,
        batch: Union[Batch, ProteinBatch],
        *,
        force: bool = False,
        pos_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reuse cached edge-local frames when node positions are unchanged."""

        if pos_override is not None:
            frames = self.localize(pos_override, batch.edge_index)
            batch.f_ij = frames
            return frames

        pos = batch.pos
        edge_index = batch.edge_index
        num_edges = edge_index.size(1)

        cached_pos = getattr(batch, "_f_ij_cache_pos", None)
        cached_frames = getattr(batch, "f_ij", None)

        # Always recompute if cache is missing, stale, or forced.
        need_full_recompute = (
            force
            or cached_pos is None
            or cached_frames is None
            or cached_frames.size(0) != num_edges
            or cached_pos.shape != pos.shape
            or cached_pos.device != pos.device
        )

        detached_pos = pos.detach()

        if need_full_recompute:
            frames = self.localize(pos, edge_index)
            batch.f_ij = frames
            batch._f_ij_cache_pos = detached_pos.clone()
            return frames

        # Identify nodes whose positions have changed beyond tolerance.
        pos_delta = torch.max(torch.abs(detached_pos - cached_pos), dim=1).values
        changed_nodes = pos_delta > self._frame_update_eps

        if changed_nodes.any():
            row, col = edge_index
            edge_mask = changed_nodes[row] | changed_nodes[col]

            if edge_mask.any():
                updated_edges = edge_index[:, edge_mask]
                updated_frames = self.localize(pos, updated_edges)

                # Clone only when an in-place update is required.
                frames = cached_frames.clone()
                frames[edge_mask] = updated_frames
                batch.f_ij = frames
            else:
                frames = cached_frames
        else:
            frames = cached_frames

        # Refresh cached positions in-place to avoid reallocations.
        cached_pos.copy_(detached_pos)
        return frames

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GCPNet encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        # Centralize node positions to make them translation-invariant
        pos_centroid, batch.pos = self.centralize(batch, batch_index=batch.batch)

        # Install `h`, `chi`, `e`, and `xi` using corresponding features built by the `FeatureFactory`
        batch.h, batch.chi, batch.e, batch.xi = (
            batch.x,
            batch.x_vector_attr,
            batch.edge_attr,
            batch.edge_vector_attr,
        )

        # Craft complete local frames corresponding to each edge, reusing cached values when possible
        batch.f_ij = self._ensure_edge_frames(batch)

        # Embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # Update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi), batch.pos = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_pos=batch.pos,
            )

        # Record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # initialize encoder outputs
        encoder_outputs = {}

        # when updating node positions, decentralize updated positions to make their updates translation-equivariant
        if self.predict_node_pos:
            batch.pos = self.decentralize(
                batch, batch_index=batch.batch, entities_centroid=pos_centroid
            )
            if self.predict_node_rep:
                # prior to scalar node predictions, re-derive local frames after performing all node position updates
                _, centralized_node_pos = self.centralize(
                    batch, batch_index=batch.batch
                )
                batch.f_ij = self._ensure_edge_frames(
                    batch, force=True, pos_override=centralized_node_pos
                )
            encoder_outputs["pos"] = batch.pos  # (n, 3) -> (batch_size, 3)

        # Summarize intermediate node representations as final predictions
        out = h
        if self.predict_node_rep:
            out = self.invariant_node_projection[0](
                ScalarVector(h, chi)
            )  # e.g., GCPLayerNorm()
            out = self.invariant_node_projection[1](
                out, batch.edge_index, batch.f_ij, node_inputs=True
            )  # e.g., GCP((h, chi)) -> h'

        encoder_outputs["node_embedding"] = out
        encoder_outputs["graph_embedding"] = self.readout(
            out, batch
        )  # (n, d) -> (batch_size, d)
        return EncoderOutput(encoder_outputs)
