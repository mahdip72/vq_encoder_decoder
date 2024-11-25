import torch
import torch.nn as nn
import torch_geometric
from models.utils import merge_features, separate_features
from gcpnet.layers.structure_proj import Dim6RotStructureHead
from gcpnet.layers.transformer_stack import TransformerStack
from gcpnet.models.vqvae import PairwisePredictionHead, RegressionHead
from gcpnet.utils.misc import _normalize, batch_orientations
from gcpnet.models.gcpnet import GCPNetModel



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

        self.vqvae_decoder_channels = configs.model.vqvae.decoder.dimension
        self.decoder_channels = decoder_configs.dimension

        self.num_heads = decoder_configs.num_heads
        self.num_layers = int(decoder_configs.num_blocks / 2)

        self.special_tokens = decoder_configs.special_tokens
        self.max_pae_bin = decoder_configs.max_pae_bin

        self.direction_loss_bins = decoder_configs.direction_loss_bins
        self.pae_bins = decoder_configs.pae_bins

        self.embed = nn.Linear(
            self.vqvae_decoder_channels, self.decoder_channels, bias=False
        )
        self.decoder_stack = TransformerStack(
            self.decoder_channels, self.num_heads, 1, self.num_layers, scale_residue=False, n_layers_geom=0
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
            # self.pae_bins,  # predicted aligned error
        ]
        self.pairwise_classification_head = PairwisePredictionHead(
            self.decoder_channels,
            downproject_dim=128,
            hidden_dim=128,
            n_bins=sum(self.pairwise_bins),
            bias=False,
        )

        # self.plddt_head = RegressionHead(
        #     embed_dim=self.decoder_channels,
        #     output_dim=configs.model.vqvae.decoder.plddt_bins,
        # )

        self.inverse_folding_head = RegressionHead(
            embed_dim=self.decoder_channels,
            output_dim=24,  # NOTE: 20 standard + 4 non-standard amino acid types, unknown type not included
        )

    def forward(
            self,
            structure_tokens: torch.Tensor,
            mask: torch.Tensor,
            batch_indices: torch.Tensor,  # NOTE: Currently unused
            x_slice_index: torch.Tensor,  # NOTE: Currently unused
    ):
        sequence_id = mask.bool()

        # not supported for now
        chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        # # check that BOS and EOS are set correctly
        # assert (
        #     structure_tokens[:, 0].eq(self.special_tokens["BOS"]).all()
        # ), "First token in structure_tokens must be BOS token"
        # assert (
        #     structure_tokens[
        #         torch.arange(structure_tokens.shape[0]), mask.sum(1) - 1
        #     ]
        #     .eq(self.special_tokens["EOS"])
        #     .all()
        # ), "Last token in structure_tokens must be EOS token"
        # assert (
        #     (structure_tokens < 0).sum() == 0
        # ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"

        x = self.embed(structure_tokens)
        # !!! NOTE: Attention mask is actually unused here so watch out
        x, _ = self.decoder_stack.forward(
            x, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id
        )

        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(mask)
        )

        # plddt_value, ptm, pae = None, None, None
        pairwise_logits = self.pairwise_classification_head(x)
        # _, _, pae_logits = [
        #     (o if o.numel() > 0 else None)
        #     for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        # ]
        dist_loss_logits, dir_loss_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ]

        # special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
        # pae = compute_predicted_aligned_error(
        #     pae_logits,  # type: ignore
        #     aa_mask=~special_tokens_mask,
        #     sequence_id=sequence_id,
        #     max_bin=self.max_pae_bin,
        # )
        # # NOTE: This might be broken for chainbreak tokens? We might align to the chainbreak
        # ptm = compute_tm(
        #     pae_logits,  # type: ignore
        #     aa_mask=~special_tokens_mask,
        #     max_bin=self.max_pae_bin,
        # )

        # plddt_logits = self.plddt_head(x)
        # plddt_value = CategoricalMixture(
        #     plddt_logits, bins=plddt_logits.shape[-1]
        # ).mean()

        # return dict(
        #     tensor7_affine=tensor7_affine,
        #     bb_pred=bb_pred,
        #     plddt=plddt_value,
        #     ptm=ptm,
        #     predicted_aligned_error=pae,
        # )

        seq_logits = self.inverse_folding_head(x)

        return bb_pred.flatten(-2), dir_loss_logits, dist_loss_logits, seq_logits