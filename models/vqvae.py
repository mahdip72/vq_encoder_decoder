import torch.nn as nn
import torch
# from vector_quantize_pytorch import VectorQuantize
from ndlinear import NdLinear


class VQVAETransformer(nn.Module):
    def __init__(self, latent_dim, codebook_size, decay, configs):
        super(VQVAETransformer, self).__init__()

        self.max_length = configs.model.max_length
        self.use_ndlinear = getattr(configs.model, 'use_ndlinear', False)

        # Define the number of residual blocks for encoder and decoder
        self.num_encoder_blocks = configs.model.vqvae.encoder.num_blocks
        self.num_decoder_blocks = configs.model.vqvae.decoder.num_blocks
        self.encoder_dim = configs.model.vqvae.encoder.dimension
        self.decoder_dim = configs.model.vqvae.decoder.dimension

        # input_shape = configs.model.struct_encoder.model_cfg.h_hidden_dim
        input_shape = 128

        # Encoder
        if self.use_ndlinear:
            self.encoder_tail = NdLinear(
                input_dims=(input_shape, 1), 
                hidden_size=(self.encoder_dim, 1)
            )
        else:
            self.encoder_tail = nn.Sequential(
                nn.Conv1d(input_shape, self.encoder_dim, kernel_size=1),
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim, nhead=configs.model.vqvae.encoder.num_heads, dim_feedforward=self.encoder_dim * 4, activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_blocks)

        self.pos_embed_encoder = nn.Parameter(torch.randn(1, self.max_length, latent_dim) * .02)

        if self.use_ndlinear:
            self.encoder_head = NdLinear(
                input_dims=(self.encoder_dim, 1),
                hidden_size=(latent_dim, 1)
            )
        else:
            self.encoder_head = nn.Sequential(
                nn.Conv1d(self.encoder_dim, latent_dim, 1),
            )

        # Vector Quantizer layer
        # self.vector_quantizer = VectorQuantize(
        #     dim=latent_dim,
        #     codebook_size=codebook_size,
        #     decay=decay,
        #     commitment_weight=configs.model.vqvae.vector_quantization.commitment_weight,
        # orthogonal_reg_weight=10,  # in paper, they recommended a value of 10
        # orthogonal_reg_max_codes=512,
        # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
        # orthogonal_reg_active_codes_only=False
        # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
        # )

        # self.pos_embed_decoder = nn.Parameter(torch.randn(1, self.max_length, self.decoder_dim) * .02)

        if self.use_ndlinear:
            self.decoder_tail = NdLinear(
                input_dims=(latent_dim, 1),
                hidden_size=(self.decoder_dim, 1)
            )
        else:
            self.decoder_tail = nn.Sequential(
                nn.Conv1d(latent_dim, self.decoder_dim, 1),
            )

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim, nhead=configs.model.vqvae.decoder.num_heads, dim_feedforward=self.decoder_dim * 4, activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=self.num_decoder_blocks)

        if self.use_ndlinear:
            self.decoder_head = NdLinear(
                input_dims=(self.decoder_dim, 1),
                hidden_size=(latent_dim, 1)  # Changed from 9 to latent_dim (768)
            )
        else:
            self.decoder_head = nn.Sequential(
                nn.Conv1d(self.decoder_dim, 9, 1),
            )

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding

    def forward(self, x, mask, return_vq_only=False):
        # Apply input projection
        if self.use_ndlinear:
            # For NdLinear we need to reshape the tensor properly
            batch_size, seq_len, channels = x.size()
            
            # Transpose to (batch, channels, seq_len) then reshape for NdLinear
            x_reshaped = x.transpose(1, 2).reshape(batch_size * seq_len, channels, 1)
            
            # Apply encoder_tail NdLinear
            x_reshaped = self.encoder_tail(x_reshaped)
            
            # Reshape back and transpose
            x = x_reshaped.reshape(batch_size, seq_len, self.encoder_dim).transpose(1, 2)
        else:
            # Original Conv1d approach
            x = x.permute(0, 2, 1)
            x = self.encoder_tail(x)

        x = x.permute(0, 2, 1)
        # Apply positional encoding to encoder
        x = x + self.pos_embed_encoder
        x = self.encoder_blocks(x)
        x = x.permute(0, 2, 1)

        if self.use_ndlinear:
            # Reshape for NdLinear
            batch_size, channels, seq_len = x.size()
            x_reshaped = x.reshape(batch_size * seq_len, channels, 1)
            
            # Apply encoder_head NdLinear
            x_reshaped = self.encoder_head(x_reshaped)
            
            # Reshape back
            x = x_reshaped.reshape(batch_size, seq_len, -1).transpose(1, 2)
        else:
            x = self.encoder_head(x)

        # x = x.permute(0, 2, 1)
        # x, indices, commit_loss = self.vector_quantizer(x)
        # x = x.permute(0, 2, 1)

        # if return_vq_only:
        #     x = x.permute(0, 2, 1)
        #     return x, indices, commit_loss

        # Apply positional encoding to decoder
        if self.use_ndlinear:
            # Reshape for NdLinear
            batch_size, channels, seq_len = x.size()
            x_reshaped = x.reshape(batch_size * seq_len, channels, 1)
            
            # Apply decoder_tail NdLinear
            x_reshaped = self.decoder_tail(x_reshaped)
            
            # Reshape back
            x = x_reshaped.reshape(batch_size, seq_len, -1).transpose(1, 2)
        else:
            x = self.decoder_tail(x)

        x = x.permute(0, 2, 1)
        # x = x + self.pos_embed_decoder
        x = self.decoder_blocks(x)

        # if self.use_ndlinear:
        #     # Need to reshape for final NdLinear layer
        #     batch_size, seq_len, channels = x.size()
        #     x_reshaped = x.reshape(batch_size * seq_len, channels, 1)
        #
        #     # Apply decoder_head NdLinear
        #     x_reshaped = self.decoder_head(x_reshaped)
        #
        #     # Reshape back to original format - using latent_dim here instead of hardcoded 9
        #     x = x_reshaped.reshape(batch_size, seq_len, -1)
        # else:clear
        #     Original approach
        #     x = x.permute(0, 2, 1)
        #     x = self.decoder_head(x)
        #     x = x.permute(0, 2, 1)

        # return x, indices, commit_loss
        return x, torch.Tensor([0]).to(x.device), torch.Tensor([0]).to(x.device)
