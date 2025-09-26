import os

import torch
import torch.nn as nn
from models.vqvae import VQVAETransformer
from models.decoders import GeometricDecoder
from models.gcpnet.models.base import (
    instantiate_encoder,
    load_pretrained_encoder,
)
from utils.utils import print_trainable_parameters
from models.utils import merge_features, separate_features


class SuperModel(nn.Module):
    def __init__(self, encoder, vqvae, configs, decoder_only=False):
        super(SuperModel, self).__init__()
        self.decoder_only = decoder_only
        if not self.decoder_only:
            self.encoder = encoder
        self.vqvae = vqvae

        self.configs = configs
        self.max_length = int(configs.model.max_length)

    def forward(self, batch, **kwargs):
        """
        Forward pass through encoder, VQ layer, and decoder.

        Args:
            batch (dict): Input batch containing 'graph'.
            return_vq_layer (bool, optional): If True, skip the decoder and return the VQ layer outputs (quantized embeddings, indices, vq loss) instead of decoded outputs.
            **kwargs: Additional arguments passed to VQVAETransformer.

        Returns:
            dict: {
                'indices': Tensor of shape (B, L) with VQ indices,
                'vq_loss': VQ loss from VQ layer,
                'ntp_logits': NTP logits if NTP is enabled,
                'outputs': Decoded outputs of shape (B, L, 3),
                'dir_loss_logits': Direction loss logits if applicable,
                'dist_loss_logits': Distance loss logits if applicable
                'valid_mask': Valid mask of shape (B, L)
            }
        """
        output_dict = {}
        nan_mask = batch['nan_masks']
        mask = batch['masks']

        if not self.decoder_only:
            batch_index = batch['graph'].batch

            if self.configs.model.encoder.name == "gcpnet":
                if self.configs.model.encoder.pretrained.enabled:
                    # No explicit casting or autocast override here.
                    model_output = self.encoder(batch['graph'])
                    x = model_output["node_embedding"]

                else:
                    _, x, _ = self.encoder(batch['graph'])

                x = separate_features(x, batch_index)

            else:
                x = self.encoder(batch, output_logits=False)

            x = merge_features(x, self.max_length)
        else:
            x = batch['indices']

        # give kwargs to vqvae
        x, indices, vq_loss, ntp_logits, valid_mask = self.vqvae(x, mask, nan_mask, **kwargs)

        output_dict["indices"] = indices
        output_dict["vq_loss"] = vq_loss
        output_dict["ntp_logits"] = ntp_logits
        output_dict["valid_mask"] = valid_mask

        if kwargs.get('return_vq_layer', False):
            output_dict["embeddings"] = x
        else:
            outputs, dir_loss_logits, dist_loss_logits = x
            output_dict["outputs"] = outputs
            output_dict["dir_loss_logits"] = dir_loss_logits
            output_dict["dist_loss_logits"] = dist_loss_logits


        return output_dict


def prepare_model(configs, logger, *, log_details=None, **kwargs):
    if log_details is None:
        log_details = configs.model.get('log_details', True)

    if not kwargs.get("decoder_only", False):
        if configs.model.encoder.name == "gcpnet":
            encoder_config_path = os.path.join("configs", "config_gcpnet_encoder.yaml")

            if configs.model.encoder.pretrained.enabled:
                encoder = load_pretrained_encoder(
                    encoder_config_path,
                    checkpoint_path=configs.model.encoder.pretrained.checkpoint_path,
                )
            else:
                components, _ = instantiate_encoder(encoder_config_path)
                encoder = components.encoder
        else:
            raise ValueError("Invalid encoder model specified!")

        if configs.model.encoder.get('freeze_parameters', False):
            for param in encoder.parameters():
                param.requires_grad = False
            if log_details:
                logger.info("Encoder parameters frozen.")

        if log_details:
            print_trainable_parameters(encoder, logger, 'Encoder')

    else:
        encoder = None

    if configs.model.vqvae.decoder.name == "geometric_decoder":
        decoder = GeometricDecoder(configs, decoder_configs=kwargs["decoder_configs"])
    else:
        raise ValueError("Invalid decoder model specified!")

    if configs.model.vqvae.decoder.get('freeze_parameters', False):
        for param in decoder.parameters():
            param.requires_grad = False
        if log_details:
            logger.info("Decoder parameters frozen.")

    vqvae = VQVAETransformer(
        decoder=decoder,
        configs=configs,
        logger=logger,
        decoder_only=kwargs.get("decoder_only", False),
    )

    if log_details:
        if not kwargs.get("decoder_only", False):
            print_trainable_parameters(nn.ModuleList([vqvae.encoder_tail, vqvae.encoder_blocks, vqvae.encoder_head]), logger, 'VQVAE Encoder')
        if vqvae.vqvae_enabled:
            print_trainable_parameters(vqvae.vector_quantizer, logger, 'VQ Layer')

        print_trainable_parameters(vqvae.decoder, logger, 'VQVAE Decoder')

        print_trainable_parameters(vqvae, logger, 'VQVAE')

    vqvae = SuperModel(encoder, vqvae, configs, decoder_only=kwargs.get("decoder_only", False))

    if log_details:
        print_trainable_parameters(vqvae, logger, 'SuperVQVAE')

    return vqvae


def compile_non_gcp_and_exclude_vq(net: nn.Module, mode = "reduce-overhead", backend: str | None = "eager") -> nn.Module:
    """
    Compile non-GCP parts of the SuperModel while excluding the VectorQuantizer.

    - Does NOT compile `net.encoder` (GCPNet or pretrained BenchMarkModel).
    - Compiles VQVAE submodules except `vector_quantizer`.
    """
    if not hasattr(net, 'vqvae'):
        return net

    # Encoder tail / blocks / head
    if hasattr(net.vqvae, 'encoder_tail') and net.vqvae.encoder_tail is not None:
        net.vqvae.encoder_tail = torch.compile(net.vqvae.encoder_tail, mode=mode, backend=backend)
    if hasattr(net.vqvae, 'encoder_blocks') and net.vqvae.encoder_blocks is not None:
        net.vqvae.encoder_blocks = torch.compile(net.vqvae.encoder_blocks, mode=mode, backend=backend)
    if hasattr(net.vqvae, 'ntp_blocks') and net.vqvae.ntp_enabled:
        net.vqvae.ntp_blocks = torch.compile(net.vqvae.ntp_blocks, mode=mode, backend=backend)
    if hasattr(net.vqvae, 'ntp_projector_head') and net.vqvae.ntp_enabled:
        net.vqvae.ntp_projector_head = torch.compile(net.vqvae.ntp_projector_head, mode=mode, backend=backend)
    if hasattr(net.vqvae, 'encoder_head') and net.vqvae.encoder_head is not None:
        net.vqvae.encoder_head = torch.compile(net.vqvae.encoder_head, mode=mode, backend=backend)

    # Exclude vector_quantizer on purpose
    # if hasattr(net.vqvae, 'vector_quantizer'):  # do not compile
    #     pass

    # Decoder
    if hasattr(net.vqvae, 'decoder') and net.vqvae.decoder is not None:
        net.vqvae.decoder = torch.compile(net.vqvae.decoder, mode=mode, backend=backend)

    return net


def compile_gcp_encoder(net: nn.Module, mode: str | None = "reduce-overhead", backend: str | None = "inductor") -> nn.Module:
    """Compile the GCP encoder component of ``SuperModel`` while leaving VQVAE untouched."""

    if not hasattr(net, "encoder") or net.encoder is None:
        return net

    def _compile(module: nn.Module) -> nn.Module:
        return torch.compile(module, mode=mode, backend=backend)

    encoder_module = net.encoder

    # Handle pretrained wrapper that contains both featuriser and encoder
    inner_encoder = getattr(encoder_module, "encoder", None)
    if isinstance(inner_encoder, nn.Module):
        encoder_module.encoder = _compile(inner_encoder)
        return net

    net.encoder = _compile(encoder_module)
    return net


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils.utils import load_configs, get_dummy_logger
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from data.dataset import custom_collate, GCPNetDataset

    config_path = "../configs/config_gcpnet.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    test_accelerator = Accelerator()

    test_model = prepare_model(test_configs, test_logger)

    print("Model loaded successfully!")

    dataset = GCPNetDataset(test_configs.train_settings.data_path,
                            seq_mode=test_configs.model.struct_encoder.use_seq.seq_embed_mode,
                            use_rotary_embeddings=test_configs.model.struct_encoder.use_rotary_embeddings,
                            use_foldseek=test_configs.model.struct_encoder.use_foldseek,
                            use_foldseek_vector=test_configs.model.struct_encoder.use_foldseek_vector,
                            top_k=test_configs.model.struct_encoder.top_k,
                            num_rbf=test_configs.model.struct_encoder.num_rbf,
                            num_positional_embeddings=test_configs.model.struct_encoder.num_positional_embeddings,
                            configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=test_configs.train_settings.batch_size, num_workers=0, pin_memory=True,
                             collate_fn=custom_collate)
    struct_embeddings = []
    test_model.eval()
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        graph = batch["graph"]
        output, _, _ = test_model(batch)
        print(output.shape)
        break
