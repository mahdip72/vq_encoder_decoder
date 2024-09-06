import torch

import torch.nn as nn

from beartype import beartype as typechecker
from functools import partial
from jaxtyping import jaxtyped
from torch_geometric.data import Batch
from typing import List, Optional, Tuple

from gcpnet import gcp
from gcpnet.utils import centralize, decentralize, get_aggregation, localize
from gcpnet.wrappers import ScalarVector
    

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
        backbone_key: str = "x_bb",
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
        See `configs/config_gcpnet.yaml` for a full list of all available arguments.

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
        :param backbone_key: Key to access the backbone node positions
            (default: ``"x_bb"``)
        :type backbone_key: str
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

        configs = kwargs["configs"]

        self.predict_backbone_positions = module_cfg.predict_backbone_positions
        self.predict_node_rep = module_cfg.predict_node_rep

        # Feature dimensionalities
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = ScalarVector(model_cfg.h_input_dim, model_cfg.chi_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)

        if configs.model.struct_encoder.use_rotary_embeddings:
            if configs.model.struct_encoder.rotary_mode == 3:
                edge_input_dims += (8, 0)  # 8+2+3+3 only for mode ==3 add 8D pos_embeddings
            else:
                edge_input_dims += (2, 0)  # 8+2
        elif configs.model.struct_encoder.use_positional_embeddings:
            edge_input_dims += (
                configs.model.struct_encoder.num_positional_embeddings, 0
            )  # 8+num_positional_embeddings

        if configs.model.struct_encoder.use_foldseek:
            node_input_dims += (10, 0)  # foldseek has 10 more node scalar features

        if configs.model.struct_encoder.use_foldseek_vector:
            node_input_dims += (0, 6)  # foldseek_vector has 6 more node vector features

        # Sequence options
        self.use_seq = configs.model.struct_encoder.use_seq.enable
        self.seq_embed_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
        self.seq_embed_dim = configs.model.struct_encoder.use_seq.seq_embed_dim

        if self.use_seq:
            node_input_dims += (self.seq_embed_dim, 0)
            if self.seq_embed_mode == "embedding":
                self.seq_embedding = nn.Embedding(20, self.seq_embed_dim)

        # Position-wise operations
        self.centralize = partial(centralize, key="x")
        self.localize = partial(localize, norm_pos_diff=module_cfg.norm_pos_diff)
        self.decentralize = partial(decentralize, key=backbone_key if self.predict_backbone_positions else "x")

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
        return ["edge_index", "x", "x_bb", "h", "chi", "e", "xi", "seq", "batch"]

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Batch, esm2_representation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Implements the forward pass of the GCPNet encoder.

        Returns the graph embedding and node (i.e., residue) embeddings in a tuple.

        :param batch: Batch of data to encode.
        :type batch: Batch
        :param esm2_representation: ESM2 representation of the protein sequence.
        :type esm2_representation: Optional[torch.Tensor]
        :return: Dictionary of graph and node embeddings. Contains
            ``graph_embedding`` and ``node_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings. Also contains the predicted node
            positions if requested.
        :rtype: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        """
        # Centralize node positions to make them translation-invariant
        pos_centroid, batch.x = self.centralize(batch, batch_index=batch.batch)

        # Craft complete local frames corresponding to each edge
        batch.f_ij = self.localize(batch.x, batch.edge_index)

        # Decide which sequence representation to use
        if self.use_seq and self.seq_embed_mode != "ESM2":
            seq = batch.seq
        elif self.use_seq and self.seq_embed_mode == "ESM2":
            seq = esm2_representation
        else:
            seq = None

        if seq is not None:
            if len(seq.shape) == 1:
                seq = self.seq_embedding(seq)
            # NOTE: A sequence representation from ESM2, PhysicsPCA, or Atchleyfactor
            batch.h = torch.cat([batch.h, seq], dim=-1)

        # Embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # Update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi), batch.x_bb = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_pos=batch.x_bb,
            )

        # Record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # When updating node positions, decentralize updated positions to make their updates translation-equivariant
        pos = None
        if self.predict_backbone_positions:
            batch.x_bb = self.decentralize(
                batch, batch_index=batch.batch, entities_centroid=pos_centroid
            )
            if self.predict_node_rep:
                # Prior to scalar node predictions, re-derive local frames after performing all node position updates
                _, centralized_node_pos = self.centralize(
                    batch, batch_index=batch.batch
                )
                batch.f_ij = self.localize(centralized_node_pos, batch.edge_index)
            pos = batch.x_bb  # (n, 3) -> (batch_size, 3)

        # Summarize intermediate node representations as final predictions
        out = h
        if self.predict_node_rep:
            out = self.invariant_node_projection[0](
                ScalarVector(h, chi)
            )  # e.g., GCPLayerNorm()
            out = self.invariant_node_projection[1](
                out, batch.edge_index, batch.f_ij, node_inputs=True
            )  # e.g., GCP((h, chi)) -> h'

        graph_feature_embedding = self.readout(out, batch.batch)  # (n, d) -> (batch_size, d)
        residue_feature_embedding = out  # (n, d)

        return graph_feature_embedding, residue_feature_embedding, pos
