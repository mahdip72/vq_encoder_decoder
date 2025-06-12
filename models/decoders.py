import torch
import torch.nn as nn
import torch_geometric
from models.utils import merge_features, separate_features
from gcpnet.layers.structure_proj import Dim6RotStructureHead
from gcpnet.models.vqvae import PairwisePredictionHead, RegressionHead
from gcpnet.utils.misc import _normalize, batch_orientations
from gcpnet.models.gcpnet import GCPNetModel
from ndlinear import NdLinear
from x_transformers import ContinuousTransformerWrapper, Encoder


class GCPNetDecoder(nn.Module):
    def __init__(self, configs, decoder_configs):
        super(GCPNetDecoder, self).__init__()
        self.max_length = configs.model.max_length

        # Define the number of residual blocks for encoder and decoder
        # self.num_encoder_blocks = configs.model.vqvae.encoder.num_blocks
        self.num_decoder_blocks = decoder_configs.num_blocks
        # self.encoder_dim = configs.model.vqvae.encoder.dimension
        self.decoder_dim = decoder_configs.dimension

        self.top_k = decoder_configs.top_k

        self.chi_init_dim = decoder_configs.chi_init_dimension
        self.xi_init_dim = decoder_configs.xi_init_dimension

        self.pos_scale_factor = decoder_configs.pos_scale_factor

        # GCPNet output (positions) projection #
        decoder_configs.module_cfg.predict_backbone_positions = True
        decoder_configs.module_cfg.predict_node_rep = False
        decoder_configs.model_cfg.num_layers = 1

        # NOTE: To preserve roto-translation invariance, only a linear term must be used
        self.output_project_init = nn.Linear(self.decoder_dim, 3 * 3, bias=False)

        output_projection_layers = []

        # Embedding layer
        decoder_configs.use_rotary_embeddings = False
        decoder_configs.use_positional_embeddings = False

        decoder_configs.use_foldseek = False
        decoder_configs.use_foldseek_vector = False

        decoder_configs.model_cfg.h_input_dim = self.decoder_dim
        decoder_configs.model_cfg.chi_input_dim = self.chi_init_dim
        decoder_configs.model_cfg.e_input_dim = self.decoder_dim * 2 + decoder_configs.module_cfg.num_rbf
        decoder_configs.model_cfg.xi_input_dim = self.xi_init_dim

        output_projection_layers.append(
            GCPNetModel(
                module_cfg=decoder_configs.module_cfg,
                model_cfg=decoder_configs.model_cfg,
                layer_cfg=decoder_configs.layer_cfg,
                configs=configs,
                backbone_key="x_bb",
            )
        )

        # Output projection layers
        decoder_configs.model_cfg.h_input_dim = decoder_configs.model_cfg.h_hidden_dim
        decoder_configs.model_cfg.chi_input_dim = self.chi_init_dim
        decoder_configs.model_cfg.e_input_dim = decoder_configs.model_cfg.h_hidden_dim * 2 + decoder_configs.module_cfg.num_rbf
        decoder_configs.model_cfg.xi_input_dim = self.xi_init_dim

        output_projection_layers.extend(
            [
                GCPNetModel(
                    module_cfg=decoder_configs.module_cfg,
                    model_cfg=decoder_configs.model_cfg,
                    layer_cfg=decoder_configs.layer_cfg,
                    configs=configs,
                    backbone_key="x_bb",
                )
                for _ in range(
                decoder_configs.model_cfg.num_bb_update_layers
            )
            ]
        )

        self.output_projections = nn.ModuleList(output_projection_layers)

    def construct_learnable_initial_graph_batch(self, feats, mask, batch_indices, x_slice_index):
        batch_num_nodes = mask.sum().item()
        device = feats.device

        h = feats[mask]
        mask = torch.ones((batch_num_nodes,), device=device, dtype=torch.bool)

        # Initialize the backbone positions in a translation-invariant manner by
        # subtracting the mean of the Ca atom positions for each corresponding residue
        x_bb = self.output_project_init(h).view(-1, 3, 3)
        x_bb = x_bb - x_bb[:, 1:2, :].mean(dim=0, keepdim=True)
        x = x_bb[:, 1, :]

        chi = batch_orientations(x.detach(), x_slice_index)

        edge_index = torch_geometric.nn.knn_graph(
            x.detach(),
            self.top_k,
            batch=batch_indices,
            loop=False,
            flow="source_to_target",
            cosine=False,
        )

        e = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=-1)
        xi = _normalize(x[edge_index[0]] - x[edge_index[1]]).unsqueeze(-2).detach()

        batch = torch_geometric.data.Batch(
            x=x,
            x_bb=x_bb,
            seq=None,
            h=h,
            chi=chi,
            e=e,
            xi=xi,
            edge_index=edge_index,
            mask=mask,
            batch=batch_indices,
            x_slice_index=x_slice_index,
        )

        return batch

    def construct_updated_graph_batch(self, batch):
        x = batch.x_bb[:, 1]

        chi = batch_orientations(x.detach(), batch.x_slice_index)

        edge_index = torch_geometric.nn.knn_graph(
            x.detach(),
            self.top_k,
            batch=batch.batch,
            loop=False,
            flow="source_to_target",
            cosine=False,
        )

        e = torch.cat([batch.h[edge_index[0]], batch.h[edge_index[1]]], dim=-1)
        xi = _normalize(x[edge_index[0]] - x[edge_index[1]]).unsqueeze(-2).detach()

        new_batch = torch_geometric.data.Batch(
            x=x,
            x_bb=batch.x_bb,
            seq=None,
            h=batch.h,
            chi=chi,
            e=e,
            xi=xi,
            edge_index=edge_index,
            mask=batch.mask,
            batch=batch.batch,
            x_slice_index=batch.x_slice_index,
        )

        return new_batch

    def forward(self, x, mask, batch_indices, x_slice_index):
        # Apply output projections in a sparse graph format
        batch = self.construct_learnable_initial_graph_batch(x, mask, batch_indices, x_slice_index)
        _, batch.h, batch.x_bb = self.output_projections[0](batch)

        for proj in self.output_projections[1:]:
            # Update the backbone positions iteratively in a translation-invariant manner by
            # subtracting the mean of the Ca atom positions for each corresponding residue
            batch.x_bb = batch.x_bb - batch.x_bb[:, 1:2, :].mean(dim=0, keepdim=True)
            batch = self.construct_updated_graph_batch(batch)
            _, batch.h, batch.x_bb = proj(batch)

        # Pad the output back into the original shape
        batch.x_bb = batch.x_bb - batch.x_bb[:, 1:2, :].mean(dim=0, keepdim=True)
        x_list = separate_features(batch.x_bb.view(-1, 9) * self.pos_scale_factor, batch.batch)
        x, *_ = merge_features(x_list, self.max_length)

        return x, None, None, None


class GeometricDecoder(nn.Module):
    def __init__(self, configs, decoder_configs):
        super(GeometricDecoder, self).__init__()

        self.max_length = configs.model.max_length

        self.use_ndlinear = getattr(configs.model, 'use_ndlinear', False)
        self.vqvae_dimension = configs.model.vqvae.vector_quantization.dim
        self.decoder_channels = decoder_configs.dimension

        self.direction_loss_bins = decoder_configs.direction_loss_bins

        # Store the decoder output scaling factor
        self.decoder_output_scaling_factor = configs.model.decoder_output_scaling_factor

        # Use either NdLinear or nn.Linear based on the flag
        if self.use_ndlinear:
            self.projector_in = NdLinear(
                input_dims=(self.max_length, self.vqvae_dimension),
                hidden_size=(self.max_length, self.decoder_channels),
            )
        else:
            self.projector_in = nn.Linear(
                self.vqvae_dimension, self.decoder_channels, bias=False
            )

        self.decoder_stack = ContinuousTransformerWrapper(
            dim_in=decoder_configs.dimension,
            dim_out=decoder_configs.dimension,
            max_seq_len=configs.model.max_length,
            num_memory_tokens=decoder_configs.num_memory_tokens,
            attn_layers=Encoder(
                dim=decoder_configs.dimension,
                ff_mult=decoder_configs.ff_mult,
                ff_glu=True,  # gate-based feed-forward (GLU family)
                ff_swish=True,  # use Swish instead of GELU → SwiGLU
                ff_no_bias=True,  # removes the two Linear biases in SwiGLU / MLP
                depth=decoder_configs.depth,
                heads=decoder_configs.heads,
                rotary_pos_emb=decoder_configs.rotary_pos_emb,
                attn_flash=decoder_configs.attn_flash,
                attn_kv_heads=decoder_configs.attn_kv_heads,
                attn_qk_norm=decoder_configs.qk_norm,
                pre_norm=decoder_configs.pre_norm,
                residual_attn=decoder_configs.residual_attn,
            )
        )

        self.affine_output_projection = Dim6RotStructureHead(
            self.decoder_channels,
            # trans_scale_factor=configs.model.struct_encoder.pos_scale_factor,
            trans_scale_factor=decoder_configs.pos_scale_factor,
            predict_torsion_angles=False,
        )

        self.pairwise_bins = [
            64,  # distogram
            self.direction_loss_bins * 6,  # direction bins
        ]
        self.pairwise_classification_head = PairwisePredictionHead(
            self.decoder_channels,
            downproject_dim=128,
            hidden_dim=128,
            n_bins=sum(self.pairwise_bins),
            bias=False,
        )

        self.inverse_folding_head = RegressionHead(
            embed_dim=self.decoder_channels,
            output_dim=24,  # NOTE: 20 standard + 4 non-standard amino acid types + 1 padding
        )

    def forward(
            self,
            structure_tokens: torch.Tensor,
            mask: torch.Tensor,
    ):
        # Apply projector_in with appropriate reshaping for NdLinear if needed
        if self.use_ndlinear:
            # Apply NdLinear projector
            x = self.projector_in(structure_tokens)
        else:
            # Original linear approach
            x = self.projector_in(structure_tokens)

        x = self.decoder_stack(x, mask=mask)

        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(mask)
        )

        # plddt_value, ptm, pae = None, None, None
        pairwise_logits = self.pairwise_classification_head(x)

        dist_loss_logits, dir_loss_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ]

        seq_logits = self.inverse_folding_head(x)

        return bb_pred.flatten(-2)*self.decoder_output_scaling_factor, dir_loss_logits, dist_loss_logits, seq_logits
