import torch.nn as nn
import torch
import math
from typing import Optional, Tuple
from x_transformers import ContinuousTransformerWrapper, Encoder
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
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

        tik_tok_cfg = getattr(
            configs.model.vqvae.vector_quantization, "tik_tok", None
        ) or {}
        self.tik_tok_enabled = tik_tok_cfg.get("enabled", False)
        compression_factor = tik_tok_cfg.get("compression_factor", 1)
        self.tik_tok_compression_factor = int(compression_factor)
        self.residual_depth = int(tik_tok_cfg.get("residual_depth", 1))
        self.use_residual_vq = self.tik_tok_enabled and self.residual_depth > 1
        if self.tik_tok_enabled:
            if self.tik_tok_compression_factor <= 0:
                raise ValueError("TikTok compression_factor must be a positive integer")
            if (self.tik_tok_compression_factor != 1) and (self.tik_tok_compression_factor % 2) != 0:
                raise ValueError("TikTok compression_factor must be an even integer")
            self.latent_token_count = math.ceil(
                self.max_length / self.tik_tok_compression_factor
            )
        else:
            self.latent_token_count = 0

        encoder_sequence_extension = self.latent_token_count if self.tik_tok_enabled else 0
        self.encoder_max_seq_len = self.max_length + encoder_sequence_extension

        if getattr(configs.train_settings.losses, "next_token_prediction", False):
            self.ntp_enabled = configs.train_settings.losses.next_token_prediction.enabled
            self.ntp_depth = getattr(configs.train_settings.losses, "next_token_prediction", 0).get('blocks', 0)
        else:
            self.ntp_enabled = False

        # input_shape = configs.model.struct_encoder.model_cfg.h_hidden_dim
        input_shape = 128

        self.encoder_causal = getattr(configs.model.vqvae.encoder, 'causal', False)
        if self.tik_tok_enabled and self.encoder_causal:
            raise ValueError(
                "TikTok latent tokens require a non-causal encoder. "
                "Disable configs.model.vqvae.encoder.causal when tik_tok is enabled."
            )

        self.tik_tok_latent_tokens = None
        self.tik_tok_padding_classifier = None

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
                max_seq_len=self.encoder_max_seq_len,
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
                self.ntp_projector_head = nn.Linear(configs.model.vqvae.vector_quantization.dim, self.codebook_size)
                if self.ntp_depth > 0:
                    self.ntp_blocks = ContinuousTransformerWrapper(
                        dim_in=configs.model.vqvae.vector_quantization.dim,
                        dim_out=configs.model.vqvae.vector_quantization.dim,
                        max_seq_len=self.encoder_max_seq_len if not self.tik_tok_enabled else self.latent_token_count * self.residual_depth,
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
                    input_dims=(self.encoder_max_seq_len, configs.model.vqvae.encoder.dimension),
                    hidden_size=(self.encoder_max_seq_len, self.vqvae_dim)
                )
            else:
                self.encoder_head = nn.Sequential(
                    nn.Conv1d(configs.model.vqvae.encoder.dimension, self.vqvae_dim, 1),
                )

            encoder_dim = configs.model.vqvae.encoder.dimension
            if self.tik_tok_enabled and self.latent_token_count > 0:
                self.tik_tok_latent_tokens = nn.Parameter(
                    torch.randn(self.latent_token_count, encoder_dim)
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
            vq_common_kwargs = dict(
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
                kmeans_iters=configs.model.vqvae.vector_quantization.kmeans_iters,
                stochastic_sample_codes=getattr(
                    configs.model.vqvae.vector_quantization, 'stochastic_sample_codes', False
                ),
                sample_codebook_temp=getattr(
                    configs.model.vqvae.vector_quantization, 'sample_codebook_temp', 0.1
                ),
            )

            if self.use_residual_vq:
                self.vector_quantizer = ResidualVQ(
                    dim=configs.model.vqvae.vector_quantization.dim,
                    num_quantizers=self.residual_depth,
                    codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
                    shared_codebook=True,
                    **vq_common_kwargs,
                )
            else:
                self.vector_quantizer = VectorQuantize(
                    dim=configs.model.vqvae.vector_quantization.dim,
                    codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
                    **vq_common_kwargs,
                )

            if configs.model.vqvae.vector_quantization.get('freeze_parameters', False):
                for param in self.vector_quantizer.parameters():
                    param.requires_grad = False
                logger.info("VQ layer parameters frozen.")

        self.decoder = decoder
        if self.tik_tok_enabled and getattr(self.decoder, 'decoder_causal', False):
            raise ValueError(
                "TikTok latent tokens require a non-causal decoder. "
                "Disable configs.model.vqvae.decoder.causal when tik_tok is enabled."
            )

        if self.tik_tok_enabled and self.latent_token_count > 0 and self.tik_tok_padding_classifier is None:
            self.tik_tok_padding_classifier = nn.Linear(
                self.vqvae_dim,
                self.tik_tok_compression_factor
            )

    def _project_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project GCPNet embeddings using NdLinear or Conv1d head."""
        if self.use_ndlinear:
            return self.encoder_tail(x)

        x = x.permute(0, 2, 1)
        x = self.encoder_tail(x)
        return x.permute(0, 2, 1)

    def _project_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Map encoder states to the quantizer dimension via NdLinear/Conv1d."""
        if self.use_ndlinear:
            return self.encoder_head(x)

        x = x.permute(0, 2, 1)
        x = self.encoder_head(x)
        return x.permute(0, 2, 1)

    def _apply_vector_quantization(
            self,
            quantizer_input: torch.Tensor,
            valid_mask: torch.Tensor,
            latent_mask_bool: Optional[torch.Tensor],
            active_tokens: Optional[torch.Tensor],
            sample_codebook_temp: Optional[float],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Pass encoder activations through the VQ layer, including TikTok/residual paths.

        Args:
            quantizer_input: Transformer encoder activations with optional TikTok latents
                appended at the tail when ``tik_tok`` is enabled.
            valid_mask: Standard key-padding mask for the original residues.
            latent_mask_bool: Boolean mask over TikTok latent slots; ``None`` when TikTok is
                disabled and the entire sequence should be quantized in one pass.
            active_tokens: Per-sample residue counts prior to compression, used to
                supervise the padding classifier.
            sample_codebook_temp: Optional runtime temperature override passed to the
                quantizer to modulate stochastic sampling.

        Returns:
            Tuple containing the decoder input (quantized embeddings), flattened codebook
            indices, VQ loss, updated latent mask, optional TikTok padding logits/targets,
            residual indices (when residual VQ is active), and latent-token counts.
        """

        tik_tok_padding_logits: Optional[torch.Tensor] = None
        tik_tok_padding_targets: Optional[torch.Tensor] = None
        unflatten_indices: Optional[torch.Tensor] = None
        latent_counts: Optional[torch.Tensor] = None

        valid_mask = valid_mask.to(torch.bool)
        vq_kwargs = {}
        if sample_codebook_temp is not None:
            vq_kwargs['sample_codebook_temp'] = sample_codebook_temp

        if self.tik_tok_enabled and self.latent_token_count > 0:
            latent_tokens = quantizer_input[:, self.max_length:, :]
            if latent_mask_bool is None:
                raise RuntimeError("TikTok latent mask was not created.")
            latent_mask_bool = latent_mask_bool.to(torch.bool)
            decoder_input, indices, vq_loss = self.vector_quantizer(
                latent_tokens,
                mask=latent_mask_bool,
                **vq_kwargs,
            )
            unflatten_indices = indices
            if self.use_residual_vq:
                indices = self._flatten_residual_indices(indices)
                vq_loss = vq_loss.sum(dim=-1)

            if self.tik_tok_padding_classifier:
                tik_tok_padding_logits, tik_tok_padding_targets, latent_counts = self._compute_tik_tok_padding_output(
                    decoder_input,
                    latent_mask_bool,
                    active_tokens,
                )

        else:
            decoder_input, indices, vq_loss = self.vector_quantizer(quantizer_input, mask=valid_mask, **vq_kwargs)

        return (
            decoder_input,
            indices,
            vq_loss,
            latent_mask_bool,
            tik_tok_padding_logits,
            tik_tok_padding_targets,
            unflatten_indices,
            latent_counts,
        )

    def _prepare_ntp_inputs(
        self,
        decoder_input: torch.Tensor,
        latent_mask: Optional[torch.Tensor],
        unflatten_indices: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble inputs and masks for the NTP head across TikTok/residual modes.

        When TikTok compression is active, the NTP stream should ignore the
        original residues and operate only on latent tokens (flattened across
        residual quantizers if enabled). Otherwise, it consumes the full
        decoder input with the standard validity mask.

        Returns:
            Tuple of ``(ntp_input, ntp_mask)`` aligned with the logits the
            projector will produce.
        """

        if self.tik_tok_enabled and self.latent_token_count > 0 and latent_mask is not None:
            if self.use_residual_vq:
                if unflatten_indices is None:
                    raise RuntimeError("Residual VQ requires unflattened indices for NTP inputs.")
                ntp_valid_mask = self._flatten_residual_mask(latent_mask)
                ntp_input = self._build_residual_ntp_input(unflatten_indices)
            else:
                ntp_valid_mask = latent_mask
                ntp_input = decoder_input
        else:
            ntp_valid_mask = valid_mask
            ntp_input = decoder_input

        return ntp_input, ntp_valid_mask

    def _decode_from_indices(
        self,
        indices: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Rebuild decoder inputs in decoder-only mode and adjust masks for TikTok.

        Besides the decoded embeddings, this returns the latent mask derived
        from ``-1`` padding, the possibly-updated validity mask, the NTP mask,
        optional TikTok padding logits/targets, and inferred sequence lengths
        when TikTok compression is enabled.
        """

        latent_mask_bool = (indices != -1)
        decoder_input = self.vector_quantizer.get_output_from_indices(indices)

        tik_tok_padding_logits: Optional[torch.Tensor] = None
        tik_tok_padding_targets: Optional[torch.Tensor] = None
        sequence_lengths: Optional[torch.Tensor] = None
        ntp_valid_mask = valid_mask
        updated_valid = valid_mask

        if self.tik_tok_enabled:
            tik_tok_padding_logits, tik_tok_padding_targets, latent_counts = self._compute_tik_tok_padding_output(
                decoder_input,
                latent_mask_bool,
                None,
            )
            predicted_remainder = tik_tok_padding_logits.argmax(dim=-1)
            sequence_lengths = latent_counts.to(torch.long) * self.tik_tok_compression_factor + predicted_remainder

            token_positions = torch.arange(self.max_length, device=mask_device).unsqueeze(0)
            updated_valid = token_positions < sequence_lengths.unsqueeze(1)
            ntp_valid_mask = updated_valid

        return (
            decoder_input,
            latent_mask_bool,
            updated_valid,
            ntp_valid_mask,
            sequence_lengths,
            tik_tok_padding_logits,
            tik_tok_padding_targets,
        )

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

    def forward(self, x, mask, nan_mask, sample_codebook_temp=None, **kwargs):
        """Encode inputs, quantize with the VQ layer, and decode to backbone coordinates.

        Args:
            x (torch.Tensor): Encoder embeddings or indices (decoder-only mode).
            mask (torch.Tensor): (B, L) bool tensor of valid residues.
            nan_mask (torch.Tensor): (B, L) bool mask removing NaN positions.
            sample_codebook_temp (float, optional): Per-forward override for the vector
                quantizer temperature. When ``None`` the layer uses its configured default.
            **kwargs: Additional options such as ``return_vq_layer``.
        """
        # mask, nan_mask are (B, N) bool; keep passing the key-padding mask as (B, N)
        mask_bool = mask.to(torch.bool)
        nan_mask_bool = nan_mask.to(torch.bool)
        valid = torch.logical_and(mask_bool, nan_mask_bool)
        ntp_logits = None
        ntp_valid_mask = valid
        tik_tok_padding_logits = None
        tik_tok_padding_targets = None
        sequence_lengths: Optional[torch.Tensor] = None
        active_tokens = None
        encoder_mask_bool = valid
        latent_mask_bool = None

        indices, vq_loss = torch.Tensor([0]).to(mask.device), torch.Tensor([0]).to(mask.device)

        if not self.decoder_only:
            encoder_input = self._project_encoder_input(x)

            if self.tik_tok_enabled and self.latent_token_count > 0:
                encoder_input, latent_mask_bool, encoder_mask_bool, active_tokens = self._append_tik_tok_latents(
                    encoder_input,
                    valid_mask=valid,
                    encoder_mask_bool=encoder_mask_bool,
                )

            encoder_attn_mask = None
            if self.encoder_causal:
                seq_len = encoder_input.size(1)
                encoder_attn_mask = self.create_causal_mask(seq_len, device=encoder_input.device)

            encoder_embeddings = self.encoder_blocks(encoder_input, mask=encoder_mask_bool, attn_mask=encoder_attn_mask)

            quantizer_input = self._project_encoder_output(encoder_embeddings)

            if self.vqvae_enabled:
                (
                    decoder_input,
                    indices,
                    vq_loss,
                    latent_mask_bool,
                    tik_tok_padding_logits,
                    tik_tok_padding_targets,
                    unflatten_indices,
                    latent_counts,
                ) = self._apply_vector_quantization(
                    quantizer_input,
                    valid,
                    latent_mask_bool,
                    active_tokens,
                    sample_codebook_temp,
                )

                if kwargs.get('return_vq_layer', False):
                    return (
                        {"embeddings": decoder_input}, indices, vq_loss, ntp_logits, ntp_valid_mask, tik_tok_padding_logits,
                        tik_tok_padding_targets,
                        sequence_lengths,
                    )
                if self.ntp_enabled:
                    ntp_input, ntp_valid_mask = self._prepare_ntp_inputs(decoder_input, latent_mask_bool,
                        unflatten_indices, valid)
                    ntp_logits = self.ntp_forward(ntp_input, valid=ntp_valid_mask)

                if self.tik_tok_enabled and self.tik_tok_padding_classifier is not None:
                    predicted_remainder = tik_tok_padding_logits.argmax(dim=-1)
                    sequence_lengths = latent_counts.to(
                        torch.long) * self.tik_tok_compression_factor + predicted_remainder
                    sequence_lengths = torch.min(sequence_lengths, torch.full_like(sequence_lengths, self.max_length))
                elif self.tik_tok_enabled and self.tik_tok_padding_classifier is None:
                    sequence_lengths = latent_mask_bool.sum(dim=-1)

            else:
                decoder_input = x
                if self.tik_tok_enabled:
                    # sequence_lengths = valid.sum(dim=1).to(torch.long)
                    # Error to say not implemented if tik_tok enabled without vqvae
                    raise NotImplementedError("TikTok mode requires vector quantization to be enabled.")

        else:
            indices = x
            (
                decoder_input,
                latent_mask_bool,
                valid,
                ntp_valid_mask,
                sequence_lengths,
                tik_tok_padding_logits,
                tik_tok_padding_targets,
            ) = self._decode_from_indices(indices, valid, mask.device)


        decoder_true_lengths = sequence_lengths if self.tik_tok_enabled else None

        x = self.decoder(decoder_input, valid, true_lengths=decoder_true_lengths)

        return (
            x,
            indices,
            vq_loss,
            ntp_logits,
            ntp_valid_mask,
            tik_tok_padding_logits,
            tik_tok_padding_targets,
            sequence_lengths,
        )

    def _append_tik_tok_latents(self, x: torch.Tensor, valid_mask: torch.Tensor, encoder_mask_bool: torch.Tensor):
        """Append TikTok latent tokens to the encoder stream and update masks.

        This helper handles the boilerplate for TikTok-mode: it concatenates the
        learnable latent token bank to each sequence, derives how many of those
        latents should remain active based on the number of valid input tokens,
        and merges the resulting latent mask into the encoder's key padding mask.

        Args:
            x: Tensor of shape ``(B, L, D)`` containing the encoder activations
               directly after ``encoder_tail`` (before TikTok augmentation).
            valid_mask: Bool tensor ``(B, L)`` where True marks residues that are
               both unmasked and non-NaN; used to compute the latent keep count.
            encoder_mask_bool: Bool tensor ``(B, L)`` representing the current
               key-padding mask passed to the transformer encoder; typically this
               is identical to ``valid_mask`` prior to TikTok augmentation.

        Returns:
            tuple:
                - augmented activations ``(B, L + latent_count, D)``
                - latent activation mask ``(B, latent_count)`` with True for
                  latents that should remain active
                - updated key-padding mask ``(B, L + latent_count)`` ready to be
                  forwarded to the encoder blocks
                - active token counts ``(B,)`` prior to compression

        Raises:
            RuntimeError: If the TikTok latent parameter tensor has not been
                initialised (should not happen when TikTok is enabled).
        """
        batch_size = x.size(0)
        if self.tik_tok_latent_tokens is None:
            raise RuntimeError("TikTok latent tokens are not initialized.")

        latent_tokens = self.tik_tok_latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([x, latent_tokens], dim=1)

        active_tokens = valid_mask.to(torch.int64).sum(dim=1)
        latent_keep = (active_tokens + self.tik_tok_compression_factor - 1) // self.tik_tok_compression_factor
        latent_keep = latent_keep.clamp(min=0, max=self.latent_token_count)

        latent_positions = torch.arange(
            self.latent_token_count,
            device=x.device,
            dtype=latent_keep.dtype
        ).unsqueeze(0)
        latent_mask_bool = latent_positions < latent_keep.unsqueeze(1)
        encoder_mask_bool = torch.cat([encoder_mask_bool, latent_mask_bool], dim=1)

        return x, latent_mask_bool, encoder_mask_bool, active_tokens

    def _compute_tik_tok_padding_output(
            self,
            latent_tokens: torch.Tensor,
            latent_mask_bool: torch.Tensor,
            active_tokens: Optional[torch.Tensor],
    ):
        """Infer original residue length from TikTok latents.

        Each latent token represents ``compression_factor`` residues (minus any
        padding). The last active latent therefore carries information about the
        padding remainder. This helper selects that latent, runs it through the
        TikTok padding classifier, and returns the logits over remainder classes
        along with optional supervision targets and latent counts.

        Args:
            latent_tokens: Quantized TikTok latent embeddings of shape
                ``(B, latent_count, D)``.
            latent_mask_bool: Boolean mask ``(B, latent_count)`` indicating which
                latent slots are valid.
            active_tokens: Optional tensor ``(B,)`` giving the true residue
                counts before compression. When provided, supervision targets are
                computed as ``active_tokens % compression_factor``. When absent
                (decoder-only inference), targets are ``None``.

        Returns:
            ``(logits, targets, latent_counts)`` where ``logits`` has shape
            ``(B, compression_factor)``, ``targets`` is either a remainder tensor
            or ``None``, and ``latent_counts`` records the number of active
            latent tokens per sample.
        """
        mask = latent_mask_bool.to(latent_tokens.dtype)
        active_counts = mask.sum(dim=1).clamp(min=1).to(torch.long)
        last_indices = (active_counts - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, latent_tokens.size(-1))
        last_latent = latent_tokens.gather(1, last_indices).squeeze(1)

        logits = self.tik_tok_padding_classifier(last_latent)

        targets = None
        if active_tokens is not None:
            targets = active_tokens.to(torch.long) % self.tik_tok_compression_factor

        return logits, targets, active_counts

    def _flatten_residual_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Flatten residual codes and push padding to the tail.

        Args:
            indices: Tensor ``(B, L, D)`` produced by :class:`ResidualVQ`, where
                ``B`` is the batch size, ``L`` the latent token count, and ``D``
                the residual depth. ``-1`` marks padded tokens. TikTok ensures
                masking is uniform across depths, so a padded position is padded
                for every quantizer.

        Returns:
            Tensor ``(B, L * D)`` with valid entries reordered depth-by-depth at
            the head and all ``-1`` padding collected at the end. The same
            scatter pattern is reused for flattening masks and embeddings so all
            downstream tensors stay aligned.

        Example:
            ``[[[0, 10], [1, 11], [-1, -1]]]`` → ``[[0, 1, 10, 11, -1, -1]]``
        """

        if indices.dim() != 3:
            return indices

        batch, length, depth = indices.shape
        flat_len = length * depth

        per_level = indices.permute(0, 2, 1).contiguous().view(batch, flat_len)
        level_mask = (indices[..., 0] >= 0).unsqueeze(1).expand(batch, depth, length)
        mask_flat = level_mask.reshape(batch, flat_len)

        mask_long = mask_flat.long()
        valid_pos = mask_long.cumsum(dim=1) - 1
        invalid_pos = (~mask_flat).long().cumsum(dim=1) - 1
        valid_count = mask_long.sum(dim=1, keepdim=True)

        target_pos = torch.where(mask_flat, valid_pos, valid_count + invalid_pos)

        reordered = indices.new_full(per_level.shape, -1)
        reordered.scatter_(1, target_pos, per_level)
        return reordered

    def _flatten_residual_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Flatten residual masks to match :meth:`_flatten_residual_indices`.

        Args:
            mask: Bool tensor ``(B, L)`` with ``True`` marking valid tokens per
                sequence position (shared across depths).

        Returns:
            Bool tensor ``(B, L * D)`` whose ``True`` values occupy the same
            leading slots as the flattened indices, with ``False`` values filling
            the trailing padding region.
        """

        if mask.dim() != 2:
            return mask

        batch, length = mask.shape
        depth = self.residual_depth
        flat_len = length * depth
        valid_count = mask.sum(dim=1, keepdim=True) * depth
        positions = torch.arange(flat_len, device=mask.device).unsqueeze(0)
        return positions < valid_count

    def _build_residual_ntp_input(self, residual_indices: torch.Tensor) -> torch.Tensor:
        """Return flattened embeddings aligned with residual-flattened indices.

        Args:
            residual_indices: Tensor ``(B, L, D)`` of residual code indices.

        Returns:
            Tensor ``(B, L * D, dim)`` containing the per-depth embeddings in the
            exact order produced by :meth:`_flatten_residual_indices`. Valid
            vectors occupy the front, while padded slots are zero-filled at the
            tail. This ensures the NTP logits, labels, and mask reference the same
            sequence positions element-wise.
        """

        if residual_indices.dim() != 3:
            raise ValueError("Residual indices must have shape (B, L, D).")

        codes_per_level = self.vector_quantizer.get_codes_from_indices(residual_indices)
        codes_per_level = codes_per_level.permute(1, 0, 2, 3).contiguous()
        batch, depth, length, dim = codes_per_level.shape
        embeddings = codes_per_level.reshape(batch, depth * length, dim)

        mask = (residual_indices[..., 0] >= 0)
        expanded_mask = mask.unsqueeze(1).expand(batch, depth, length)
        mask_flat = expanded_mask.reshape(batch, depth * length)

        mask_long = mask_flat.long()
        valid_pos = mask_long.cumsum(dim=1) - 1
        invalid_pos = (~mask_flat).long().cumsum(dim=1) - 1
        valid_count = mask_long.sum(dim=1, keepdim=True)

        target_pos = torch.where(mask_flat, valid_pos, valid_count + invalid_pos)

        embeddings = embeddings * mask_flat.unsqueeze(-1)
        output = embeddings.new_zeros(embeddings.shape)
        scatter_indices = target_pos.unsqueeze(-1).expand_as(embeddings)
        output.scatter_(1, scatter_indices, embeddings)
        return output
