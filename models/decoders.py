import torch
import torch.nn as nn
from models.gcpnet.layers.structure_proj import Dim6RotStructureHead
from models.gcpnet.heads import PairwisePredictionHead, RegressionHead
from ndlinear import NdLinear
from x_transformers import ContinuousTransformerWrapper, Encoder


class GeometricDecoder(nn.Module):
    def __init__(self, configs, decoder_configs):
        super(GeometricDecoder, self).__init__()

        self.max_length = configs.model.max_length
        self.decoder_causal = getattr(decoder_configs, "causal", False)

        self.use_ndlinear = getattr(configs.model, 'use_ndlinear', False)
        self.vqvae_dimension = configs.model.vqvae.vector_quantization.dim
        self.decoder_channels = decoder_configs.dimension

        self.direction_loss_bins = decoder_configs.direction_loss_bins

        # Store the decoder output scaling factor
        self.decoder_output_scaling_factor = configs.model.decoder_output_scaling_factor

        losses_cfg = configs.train_settings.losses
        self.enable_pairwise_losses = (
            losses_cfg.binned_distance_classification.enabled
            or losses_cfg.binned_direction_classification.enabled
        )

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

        if self.enable_pairwise_losses:
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
        else:
            self.pairwise_bins = []
            self.pairwise_classification_head = None

    def create_causal_mask(self, seq_len, device):
        """
        Create a lower-triangular (causal) boolean attention mask of shape (seq_len, seq_len),
        where True indicates allowed attention (token i attends only to tokens j <= i).
        """
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()

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

        decoder_attn_mask = None
        if self.decoder_causal:
            seq_len = x.size(1)
            decoder_attn_mask = self.create_causal_mask(seq_len, device=x.device)

        x = self.decoder_stack(x, mask=mask, attn_mask=decoder_attn_mask)

        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(mask)
        )

        # plddt_value, ptm, pae = None, None, None
        dist_loss_logits = None
        dir_loss_logits = None
        if self.enable_pairwise_losses:
            pairwise_logits = self.pairwise_classification_head(x)
            dist_loss_logits, dir_loss_logits = [
                (o if o.numel() > 0 else None)
                for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
            ]

        return bb_pred.flatten(-2)*self.decoder_output_scaling_factor, dir_loss_logits, dist_loss_logits
