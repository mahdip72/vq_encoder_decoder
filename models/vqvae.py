import torch.nn as nn
import torch
from x_transformers import ContinuousTransformerWrapper, Encoder
from vector_quantize_pytorch import VectorQuantize
from ndlinear import NdLinear


class VQVAETransformer(nn.Module):
    def __init__(self, configs, decoder, logger, decoder_only=False):
        super(VQVAETransformer, self).__init__()

        self.max_length = configs.model.max_length
        self.use_ndlinear = getattr(configs.model, 'use_ndlinear', False)
        self.decoder_only = decoder_only  # If True, only the decoder is used, no encoder

        # Define the number of residual blocks for encoder and decoder

        self.vqvae_enabled = configs.model.vqvae.vector_quantization.enabled
        self.vqvae_dim = configs.model.vqvae.vector_quantization.dim
        self.codebook_size = configs.model.vqvae.vector_quantization.codebook_size
        if getattr(configs.train_settings.losses, "next_token_prediction", False):
            self.ntp_enabled = configs.train_settings.losses.next_token_prediction.enabled
            self.ntp_depth = configs.train_settings.losses.next_token_prediction.blocks
        else:
            self.ntp_enabled = False

        # input_shape = configs.model.struct_encoder.model_cfg.h_hidden_dim
        input_shape = 128

        self.encoder_causal = getattr(configs.model.vqvae.encoder, 'causal', False)

        if not self.decoder_only:
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
                    ff_no_bias=True,  # removes the two Linear biases in SwiGLU / MLP
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

            # Next-token prediction head from encoder block embeddings
            if self.ntp_enabled:
                self.ntp_projector_head = nn.Linear(configs.model.vqvae.encoder.dimension, self.codebook_size)
                if self.ntp_depth > 0:
                    self.ntp_blocks = ContinuousTransformerWrapper(
                        dim_in=configs.model.vqvae.encoder.dimension,
                        dim_out=configs.model.vqvae.encoder.dimension,
                        max_seq_len=configs.model.max_length,
                        num_memory_tokens=configs.model.vqvae.encoder.num_memory_tokens,
                        attn_layers=Encoder(
                            dim=configs.model.vqvae.encoder.dimension,
                            ff_mult=configs.model.vqvae.encoder.ff_mult,
                            ff_glu=True,  # gate-based feed-forward (GLU family)
                            ff_swish=True,  # use Swish instead of GELU → SwiGLU
                            ff_no_bias=True,  # removes the two Linear biases in SwiGLU / MLP
                            depth=configs.train_settings.losses.next_token_prediction.blocks,
                            heads=configs.model.vqvae.encoder.heads,
                            rotary_pos_emb=configs.model.vqvae.encoder.rotary_pos_emb,
                            attn_flash=configs.model.vqvae.encoder.attn_flash,
                            attn_kv_heads=configs.model.vqvae.encoder.attn_kv_heads,
                            attn_qk_norm=configs.model.vqvae.encoder.qk_norm,
                            pre_norm=configs.model.vqvae.encoder.pre_norm,
                            residual_attn=configs.model.vqvae.encoder.residual_attn,
                        )
                    )
                elif self.ntp_depth < 0:
                    raise ValueError("Invalid number of next-token prediction blocks specified.")

            if self.use_ndlinear:
                self.encoder_head = NdLinear(
                    input_dims=(self.max_length, configs.model.vqvae.encoder.dimension),
                    hidden_size=(self.max_length, self.vqvae_dim)
                )
            else:
                self.encoder_head = nn.Sequential(
                    nn.Conv1d(configs.model.vqvae.encoder.dimension, self.vqvae_dim, 1),
                )

            if configs.model.vqvae.encoder.get('freeze_parameters', False):
                for param in self.encoder_tail.parameters():
                    param.requires_grad = False
                for param in self.encoder_blocks.parameters():
                    param.requires_grad = False
                for param in self.encoder_head.parameters():
                    param.requires_grad = False
                logger.info("VQVAE encoder parameters frozen.")

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
                threshold_ema_dead_code=configs.model.vqvae.vector_quantization.threshold_ema_dead_code,
                kmeans_init=configs.model.vqvae.vector_quantization.kmeans_init,
                kmeans_iters=configs.model.vqvae.vector_quantization.kmeans_iters  # number of kmeans iterations to calculate the centroids for the codebook on init
            )

            if configs.model.vqvae.vector_quantization.get('freeze_parameters', False):
                for param in self.vector_quantizer.parameters():
                    param.requires_grad = False
                logger.info("VQ layer parameters frozen.")

        self.decoder = decoder

    def create_causal_mask(self, seq_len, device):
        """
        Create a lower-triangular (causal) boolean attention mask of shape (seq_len, seq_len),
        where True indicates allowed attention (token i attends only to tokens j <= i).
        """
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()

    def ntp_forward(self, x, valid):

        seq_len = x.size(1)
        ntp_attn_mask = self.create_causal_mask(seq_len, device=x.device)

        if self.ntp_depth > 0:
            x = self.ntp_blocks(x, mask=valid, attn_mask=ntp_attn_mask)

        ntp_logits = self.ntp_projector_head(x)

        return ntp_logits

    def forward(self, x, mask, nan_mask, **kwargs):
        # mask, nan_mask are (B, N) bool; keep passing the key-padding mask as (B, N)
        valid = torch.logical_and(mask, nan_mask)
        ntp_logits = None
        if not self.decoder_only:
            # Apply input projection
            if self.use_ndlinear:
                # Apply encoder_tail NdLinear
                x = self.encoder_tail(x)
            else:
                # Original Conv1d approach
                x = x.permute(0, 2, 1)
                x = self.encoder_tail(x)
                x = x.permute(0, 2, 1)

            encoder_attn_mask = None
            if self.encoder_causal:
                seq_len = x.size(1)
                encoder_attn_mask = self.create_causal_mask(seq_len, device=x.device)

            x = self.encoder_blocks(x, mask=valid, attn_mask=encoder_attn_mask)

            # NTP logits from encoder block outputs
            if self.ntp_enabled:
                ntp_logits = self.ntp_forward(x, valid=valid)

            if self.use_ndlinear:
                # Apply encoder_head NdLinear
                x = self.encoder_head(x)
            else:
                x = x.permute(0, 2, 1)
                x = self.encoder_head(x)
                x = x.permute(0, 2, 1)

        indices, commit_loss = torch.Tensor([0]).to(mask.device), torch.Tensor([0]).to(mask.device)

        if self.vqvae_enabled:
            if not self.decoder_only:
                # Apply vector quantization
                x, indices, commit_loss = self.vector_quantizer(x, mask=valid)

                if kwargs.get('return_vq_layer', False):
                    return x, indices, commit_loss, ntp_logits, valid
            else:
                indices = x
                x = self.vector_quantizer.get_output_from_indices(indices)
        x = self.decoder(x, valid)

        return x, indices, commit_loss, ntp_logits, valid
