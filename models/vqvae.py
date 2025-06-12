import torch.nn as nn
import torch
from x_transformers import ContinuousTransformerWrapper, Encoder
from vector_quantize_pytorch import VectorQuantize
from ndlinear import NdLinear


class VQVAETransformer(nn.Module):
    def __init__(self, configs, decoder):
        super(VQVAETransformer, self).__init__()

        self.max_length = configs.model.max_length
        self.use_ndlinear = getattr(configs.model, 'use_ndlinear', False)

        # Define the number of residual blocks for encoder and decoder

        self.vqvae_enabled = configs.model.vqvae.vector_quantization.enabled
        self.vqvae_dim = configs.model.vqvae.vector_quantization.dim

        # input_shape = configs.model.struct_encoder.model_cfg.h_hidden_dim
        input_shape = 128

        # Encoder
        if self.use_ndlinear:
            self.encoder_tail = NdLinear(
                input_dims=(self.max_length, input_shape),
                hidden_size=(self.max_length, configs.model.vqvae.encoder.dimension)
            )
        else:
            self.encoder_tail = nn.Sequential(
                nn.Conv1d(input_shape, configs.model.vqvae.encoder.dimension, kernel_size=1),
            )

        self.encoder_blocks = ContinuousTransformerWrapper(
            dim_in=configs.model.vqvae.encoder.dimension,
            dim_out=configs.model.vqvae.encoder.dimension,
            max_seq_len=configs.model.max_length,
            num_memory_tokens=configs.model.vqvae.encoder.num_memory_tokens,
            attn_layers=Encoder(
                dim=configs.model.vqvae.encoder.dimension,
                ff_mult=configs.model.vqvae.encoder.ff_mult,
                ff_glu=True,  # gate-based feed-forward (GLU family)
                ff_swish=True,  # use Swish instead of GELU → SwiGLU
                no_bias=True,
                depth=configs.model.vqvae.encoder.depth,
                heads=configs.model.vqvae.encoder.heads,
                rotary_pos_emb=configs.model.vqvae.encoder.rotary_pos_emb,
                attn_flash=configs.model.vqvae.encoder.attn_flash,
                attn_kv_heads=configs.model.vqvae.encoder.attn_kv_heads,
                attn_qk_norm=configs.model.vqvae.encoder.qk_norm,
                pre_norm=configs.model.vqvae.encoder.pre_norm,
                residual_attn=configs.model.vqvae.encoder.residual_attn,
            )
        )

        if self.use_ndlinear:
            self.encoder_head = NdLinear(
                input_dims=(self.max_length, configs.model.vqvae.encoder.dimension),
                hidden_size=(self.max_length, self.vqvae_dim)
            )
        else:
            self.encoder_head = nn.Sequential(
                nn.Conv1d(configs.model.vqvae.encoder.dimension, self.vqvae_dim, 1),
            )

        # Vector Quantizer layer
        if self.vqvae_enabled:
            self.vector_quantizer = VectorQuantize(
                dim=configs.model.vqvae.vector_quantization.dim,
                codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
                decay=configs.model.vqvae.vector_quantization.decay,
                commitment_weight=configs.model.vqvae.vector_quantization.commitment_weight,
                orthogonal_reg_weight=configs.model.vqvae.vector_quantization.orthogonal_reg_weight,
                orthogonal_reg_max_codes=configs.model.vqvae.vector_quantization.orthogonal_reg_max_codes,
                # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
                orthogonal_reg_active_codes_only=configs.model.vqvae.vector_quantization.orthogonal_reg_active_codes_only,
                # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
                rotation_trick=configs.model.vqvae.vector_quantization.rotation_trick,
            )

        self.decoder = decoder

    def forward(self, x, mask, **kwargs):
        # Apply input projection
        if self.use_ndlinear:
            # Apply encoder_tail NdLinear
            x = self.encoder_tail(x)
        else:
            # Original Conv1d approach
            x = x.permute(0, 2, 1)
            x = self.encoder_tail(x)
            x = x.permute(0, 2, 1)

        x = self.encoder_blocks(x, mask=mask)

        if self.use_ndlinear:
            # Apply encoder_head NdLinear
            x = self.encoder_head(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.encoder_head(x)
            x = x.permute(0, 2, 1)

        indices, commit_loss = torch.Tensor([0]).to(x.device), torch.Tensor([0]).to(x.device)

        if self.vqvae_enabled:
            # Apply vector quantization
            x, indices, commit_loss = self.vector_quantizer(x, mask=mask)

            if kwargs.get('return_vq_layer', False):
                return x, indices, commit_loss

        x = self.decoder(x, mask)
        return x, indices, commit_loss