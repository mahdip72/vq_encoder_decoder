import torch.nn as nn
import torch
from vector_quantize_pytorch import VectorQuantize
from ndlinear import NdLinear


class VQVAETransformer(nn.Module):
    def __init__(self, configs):
        super(VQVAETransformer, self).__init__()

        self.max_length = configs.model.max_length
        self.use_ndlinear = getattr(configs.model, 'use_ndlinear', False)
        self.positional_encoding_encoder = configs.model.vqvae.positional_encoding
        self.is_causal = configs.model.vqvae.causal_attention

        # Define the number of residual blocks for encoder and decoder
        self.num_encoder_blocks = configs.model.vqvae.encoder.num_blocks
        self.encoder_dim = configs.model.vqvae.encoder.dimension

        self.vqvae_enabled = configs.model.vqvae.vector_quantization.enabled
        self.vqvae_dim = configs.model.vqvae.vector_quantization.dim

        # input_shape = configs.model.struct_encoder.model_cfg.h_hidden_dim
        input_shape = 128

        # Encoder
        if self.use_ndlinear:
            self.encoder_tail = NdLinear(
                input_dims=(self.max_length, input_shape),
                hidden_size=(self.max_length, self.encoder_dim)
            )
        else:
            self.encoder_tail = nn.Sequential(
                nn.Conv1d(input_shape, self.encoder_dim, kernel_size=1),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim, nhead=configs.model.vqvae.encoder.num_heads, dim_feedforward=self.encoder_dim * 4,
            activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_blocks)

        if self.positional_encoding_encoder:
            self.pos_embed_encoder = nn.Parameter(torch.randn(1, self.max_length, self.encoder_dim) * .02)

        if self.use_ndlinear:
            self.encoder_head = NdLinear(
                input_dims=(self.max_length, self.encoder_dim),
                hidden_size=(self.max_length, self.vqvae_dim)
            )
        else:
            self.encoder_head = nn.Sequential(
                nn.Conv1d(self.encoder_dim, self.vqvae_dim, 1),
            )

        # Vector Quantizer layer
        if self.vqvae_enabled:
            self.vector_quantizer = VectorQuantize(
                dim=configs.model.vqvae.vector_quantization.dim,
                codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
                decay=configs.model.vqvae.vector_quantization.decay,
                commitment_weight=configs.model.vqvae.vector_quantization.commitment_weight,
            )

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding

    def forward(self, x, mask, return_vq_only=False):
        mask = ~mask
        # Apply input projection
        if self.use_ndlinear:
            # Apply encoder_tail NdLinear
            x = self.encoder_tail(x)
        else:
            # Original Conv1d approach
            x = x.permute(0, 2, 1)
            x = self.encoder_tail(x)
            x = x.permute(0, 2, 1)

        if self.positional_encoding_encoder:
            # Apply positional encoding to encoder
            x = x + self.pos_embed_encoder

        x = self.encoder_blocks(
            x,
            mask=torch.nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device,
                                                                      dtype=torch.bool) if self.is_causal else None,
            src_key_padding_mask=mask,
            is_causal=self.is_causal
        )

        if self.use_ndlinear:
            # Apply encoder_head NdLinear
            x = self.encoder_head(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.encoder_head(x)
            x = x.permute(0, 2, 1)

        if self.vqvae_enabled:
            # Apply vector quantization
            x = x.permute(0, 2, 1)
            x, indices, commit_loss = self.vector_quantizer(x)
            x = x.permute(0, 2, 1)

            if return_vq_only:
                x = x.permute(0, 2, 1)
                return x, indices, commit_loss

        # return x, indices, commit_loss
        return x, torch.Tensor([0]).to(x.device), torch.Tensor([0]).to(x.device)
