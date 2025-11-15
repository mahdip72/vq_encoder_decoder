import torch
import math
from typing import Tuple, Optional
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

        tik_tok_cfg = getattr(
            configs.model.vqvae.vector_quantization, "tik_tok", None
        ) or {}
        self.tik_tok_enabled = tik_tok_cfg.get("enabled", False)
        compression_factor = tik_tok_cfg.get("compression_factor", 1)
        self.tik_tok_compression_factor = int(compression_factor)
        if self.tik_tok_enabled:
            if self.tik_tok_compression_factor <= 0:
                raise ValueError("TikTok compression_factor must be a positive integer")
            if (self.tik_tok_compression_factor != 1) and (self.tik_tok_compression_factor % 2) != 0:
                raise ValueError("TikTok compression_factor must be an even integer")
            if self.decoder_causal:
                raise ValueError(
                    "TikTok latent tokens require a non-causal decoder. "
                    "Disable configs.model.vqvae.decoder.causal when tik_tok is enabled."
                )
            self.latent_token_count = math.ceil(
                self.max_length / self.tik_tok_compression_factor
            )
        else:
            self.latent_token_count = 0
        self.decoder_max_seq_len = self.max_length + (
            self.latent_token_count if self.tik_tok_enabled else 0
        )

        projector_input_length = (
            self.latent_token_count if self.tik_tok_enabled else self.max_length
        )

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
                input_dims=(projector_input_length, self.vqvae_dimension),
                hidden_size=(projector_input_length, self.decoder_channels),
            )
        else:
            self.projector_in = nn.Linear(
                self.vqvae_dimension, self.decoder_channels, bias=False
            )

        if self.tik_tok_enabled:
            self.decoder_mask_token = nn.Parameter(
                torch.randn(self.decoder_channels)
            )
        else:
            self.decoder_mask_token = None

        self.decoder_stack = ContinuousTransformerWrapper(
            dim_in=decoder_configs.dimension,
            dim_out=decoder_configs.dimension,
            max_seq_len=self.decoder_max_seq_len,
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

        inverse_folding_cfg = getattr(configs.train_settings.losses, "inverse_folding", None)
        self.enable_inverse_folding = bool(
            inverse_folding_cfg and getattr(inverse_folding_cfg, "enabled", False)
        )
        self.inverse_folding_classes = (
            getattr(inverse_folding_cfg, "num_classes", 21) if self.enable_inverse_folding else 0
        )
        self.inverse_folding_pre_decoder = False
        if self.enable_inverse_folding:
            self.inverse_folding_pre_decoder = bool(
                getattr(inverse_folding_cfg, "pre_decoder", False)
            )
            if self.inverse_folding_pre_decoder and self.tik_tok_enabled:
                raise ValueError(
                    "Inverse folding head cannot run before the decoder when TikTok latents are enabled."
                )

        plddt_cfg = getattr(configs.train_settings.losses, "plddt", None)
        self.enable_plddt = bool(
            plddt_cfg and getattr(plddt_cfg, "enabled", False)
        )
        self.plddt_pre_decoder = False
        if self.enable_plddt:
            self.plddt_pre_decoder = bool(
                getattr(plddt_cfg, "pre_decoder", True)
            )
            if self.plddt_pre_decoder and self.tik_tok_enabled:
                raise ValueError(
                    "pLDDT head cannot run before the decoder when TikTok latents are enabled."
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

        self.inverse_folding_head = (
            nn.Linear(self.decoder_channels, self.inverse_folding_classes)
            if self.enable_inverse_folding
            else None
        )
        self.plddt_head = (
            nn.Linear(self.decoder_channels, 1) if self.enable_plddt else None
        )

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
            *,
            true_lengths: Optional[torch.Tensor] = None,
    ):
        # Apply projector_in with appropriate reshaping for NdLinear if needed
        if self.use_ndlinear:
            # Apply NdLinear projector
            x = self.projector_in(structure_tokens)
        else:
            # Original linear approach
            x = self.projector_in(structure_tokens)

        decoder_mask_bool = mask.to(torch.bool)

        if true_lengths is not None:
            true_lengths = true_lengths.to(torch.long)
            positions = torch.arange(self.max_length, device=mask.device).unsqueeze(0)
            decoder_mask_bool = positions < true_lengths.unsqueeze(1)

        if self.tik_tok_enabled:
            x, decoder_mask_bool = self._build_decoder_tik_tok_stream(
                latent_tokens=x,
                original_mask=decoder_mask_bool,
            )

        decoder_attn_mask = None
        if self.decoder_causal:
            seq_len = x.size(1)
            decoder_attn_mask = self.create_causal_mask(seq_len, device=x.device)

        seq_logits = None
        if self.inverse_folding_head is not None and self.inverse_folding_pre_decoder:
            seq_logits = self.inverse_folding_head(x)

        plddt_logits = None
        if self.plddt_head is not None and self.plddt_pre_decoder:
            plddt_logits = self.plddt_head(x)

        x = self.decoder_stack(x, mask=decoder_mask_bool, attn_mask=decoder_attn_mask)

        if self.tik_tok_enabled:
            x = x[:, :self.max_length, :]

        if self.inverse_folding_head is not None and not self.inverse_folding_pre_decoder:
            seq_logits = self.inverse_folding_head(x)
        if self.plddt_head is not None and not self.plddt_pre_decoder:
            plddt_logits = self.plddt_head(x)

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

        return {
            "outputs": bb_pred.flatten(-2) * self.decoder_output_scaling_factor,
            "dir_loss_logits": dir_loss_logits,
            "dist_loss_logits": dist_loss_logits,
            "seq_logits": seq_logits,
            "plddt_logits": plddt_logits,
            "embeddings": x,
        }

    def _build_decoder_tik_tok_stream(
        self,
        latent_tokens: torch.Tensor,
        original_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct the decoder input sequence when TikTok is enabled.

        Args:
            latent_tokens: Tensor ``(B, latent_count, D)`` output by the VQ layer.
            original_mask: Bool tensor ``(B, max_length)`` indicating valid
                residue positions prior to TikTok augmentation.

        Returns:
            tuple containing:
                - Concatenated decoder input ``(B, max_length + latent_count, D)``
                - Updated key padding mask of shape ``(B, max_length + latent_count)``

        Raises:
            RuntimeError: If the decoder mask token has not been created.
        """
        if self.decoder_mask_token is None:
            raise RuntimeError("TikTok decoder mask token is not initialized.")

        batch_size = latent_tokens.size(0)
        latent_token_length = latent_tokens.size(1)

        mask_token = self.decoder_mask_token.unsqueeze(0).unsqueeze(0)
        mask_tokens = mask_token.expand(batch_size, self.max_length, -1)
        mask_tokens = mask_tokens * original_mask.to(mask_tokens.dtype).unsqueeze(-1)

        active_tokens = original_mask.to(torch.int64).sum(dim=1)
        latent_keep = (active_tokens + self.tik_tok_compression_factor - 1) // self.tik_tok_compression_factor
        latent_keep = latent_keep.clamp(min=0, max=latent_token_length)

        latent_positions = torch.arange(
            latent_token_length,
            device=latent_tokens.device,
            dtype=latent_keep.dtype
        ).unsqueeze(0)
        latent_mask_bool = latent_positions < latent_keep.unsqueeze(1)
        latent_tokens = latent_tokens * latent_mask_bool.to(latent_tokens.dtype).unsqueeze(-1)

        decoder_mask_bool = torch.cat([original_mask, latent_mask_bool], dim=1)
        decoder_input = torch.cat([mask_tokens, latent_tokens], dim=1)

        return decoder_input, decoder_mask_bool
