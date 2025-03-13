import torch.nn as nn
import torch
# from vector_quantize_pytorch import VectorQuantize



class VQVAETransformer(nn.Module):
    def __init__(self, latent_dim, codebook_size, decay, configs):
        super(VQVAETransformer, self).__init__()

        self.max_length = configs.model.max_length

        # Define the number of residual blocks for encoder and decoder
        self.num_encoder_blocks = configs.model.vqvae.encoder.num_blocks
        self.num_decoder_blocks = configs.model.vqvae.decoder.num_blocks
        self.encoder_dim = configs.model.vqvae.encoder.dimension
        self.decoder_dim = configs.model.vqvae.decoder.dimension

        # input_shape = configs.model.struct_encoder.model_cfg.h_hidden_dim
        input_shape = 128

        # Encoder
        self.encoder_tail = nn.Sequential(
            nn.Conv1d(input_shape, self.encoder_dim, kernel_size=1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim, nhead=8, dim_feedforward=self.encoder_dim * 4, activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_blocks)

        self.pos_embed_encoder = nn.Parameter(torch.randn(1, self.max_length, latent_dim) * .02)

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

        self.decoder_tail = nn.Sequential(
            nn.Conv1d(latent_dim, self.decoder_dim, 1),
        )

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim, nhead=8, dim_feedforward=self.decoder_dim * 4, activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=self.num_decoder_blocks)

        self.decoder_head = nn.Sequential(
            nn.Conv1d(self.decoder_dim, 9, 1),
        )

    @staticmethod
    def drop_positional_encoding(embedding, pos_embed):
        embedding = embedding + pos_embed
        return embedding

    def forward(self, x, mask, return_vq_only=False):
        # Apply input projection
        x = x.permute(0, 2, 1)
        x = self.encoder_tail(x)

        x = x.permute(0, 2, 1)
        # Apply positional encoding to encoder
        x = x + self.pos_embed_encoder
        x = self.encoder_blocks(x)
        x = x.permute(0, 2, 1)

        x = self.encoder_head(x)

        # x = x.permute(0, 2, 1)
        # x, indices, commit_loss = self.vector_quantizer(x)
        # x = x.permute(0, 2, 1)

        # if return_vq_only:
        #     x = x.permute(0, 2, 1)
        #     return x, indices, commit_loss

        # Apply positional encoding to decoder
        x = self.decoder_tail(x)

        x = x.permute(0, 2, 1)
        # x = x + self.pos_embed_decoder
        x = self.decoder_blocks(x)
        # x = x.permute(0, 2, 1)

        # x = self.decoder_head(x)*10
        # x = x.permute(0, 2, 1)

        # return x, indices, commit_loss
        return x, torch.Tensor([0]).to(x.device), torch.Tensor([0]).to(x.device)