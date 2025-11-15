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
            sample_codebook_temp (float, optional): Runtime override for the VQ sampling temperature;
                forwarded to the VQVAE block so different decoding schedules can be tested without
                mutating module state.
            **kwargs: Additional arguments passed to VQVAETransformer.

        Returns:
            dict containing VQ outputs, decoder predictions, and optional
            auxiliary tensors (NTP, TikTok padding classifier, masks).
        """
        output_dict = {}
        nan_mask = batch['nan_masks']
        mask = batch['masks']

        if not self.decoder_only:
            batch_index = batch['graph'].batch

            # No explicit casting or autocast override here.
            model_output = self.encoder(batch['graph'])

            x = model_output["node_embedding"]

            x = separate_features(x, batch_index)
            x = merge_features(x, self.max_length)
        else:
            x = batch['indices']

        sample_codebook_temp = kwargs.pop('sample_codebook_temp', None)

        # give kwargs to vqvae
        (
            x,
            indices,
            vq_loss,
            ntp_logits,
            ntp_mask,
            tik_tok_padding_logits,
            tik_tok_padding_targets,
            sequence_lengths,
        ) = self.vqvae(x, mask, nan_mask, sample_codebook_temp=sample_codebook_temp, **kwargs)

        output_dict["indices"] = indices
        output_dict["vq_loss"] = vq_loss
        output_dict["ntp_logits"] = ntp_logits
        output_dict["ntp_mask"] = ntp_mask
        output_dict["tik_tok_padding_logits"] = tik_tok_padding_logits
        output_dict["tik_tok_padding_targets"] = tik_tok_padding_targets
        output_dict["sequence_lengths"] = sequence_lengths

        if kwargs.get('return_vq_layer', False):
            output_dict["embeddings"] = x
        else:
            seq_logits = None
            if isinstance(x, (tuple, list)):
                if len(x) == 4:
                    outputs, dir_loss_logits, dist_loss_logits, seq_logits = x
                else:
                    outputs, dir_loss_logits, dist_loss_logits = x
            else:
                outputs, dir_loss_logits, dist_loss_logits = x
            output_dict["outputs"] = outputs
            output_dict["dir_loss_logits"] = dir_loss_logits
            output_dict["dist_loss_logits"] = dist_loss_logits
            output_dict["seq_logits"] = seq_logits


        return output_dict


def prepare_model(configs, logger, *, log_details=False, **kwargs):

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


def compile_non_gcp_and_exclude_vq(net: nn.Module, mode: str | None, backend: str | None) -> nn.Module:
    """
    Compile non-GCP parts of the SuperModel while excluding the VectorQuantizer.

    Does NOT compile `net.encoder` (GCPNet or pretrained BenchMarkModel).
    Compiles every child module of VQVAE except `vector_quantizer`.

    Parameters
    ----------
    net : nn.Module
        The SuperModel instance.
    mode : str | None
        torch.compile optimization mode. Options: "reduce-overhead", "default", "max-autotune", or None.
        (None forwards directly to torch.compile allowing its internal default.)
    backend : str | None
        torch.compile backend. Common options here: "eager" or "inductor". (None lets torch pick a default.)

    Returns
    -------
    nn.Module
        The (in-place) modified model.
    """
    if not hasattr(net, 'vqvae'):
        return net

    vqvae = net.vqvae

    def _compile_child(name: str, child):
        if child is None:
            return
        if name == 'vector_quantizer':
            return

        if isinstance(child, nn.ModuleList):
            compiled = nn.ModuleList([
                torch.compile(module, mode=mode, backend=backend)
                if isinstance(module, nn.Module) else module
                for module in child
            ])
            setattr(vqvae, name, compiled)
            return

        if isinstance(child, nn.ModuleDict):
            compiled_dict = nn.ModuleDict({
                key: torch.compile(module, mode=mode, backend=backend)
                if isinstance(module, nn.Module) else module
                for key, module in child.items()
            })
            setattr(vqvae, name, compiled_dict)
            return

        if isinstance(child, nn.Module):
            setattr(vqvae, name, torch.compile(child, mode=mode, backend=backend))

    compiled_names: set[str] = set()
    for name, child in list(vqvae.named_children()):
        _compile_child(name, child)
        compiled_names.add(name)

    # Also compile any direct attributes that are modules but not registered as children
    extra_attrs = (
        'encoder_tail', 'encoder_blocks', 'encoder_head',
        'ntp_blocks', 'ntp_projector_head', 'decoder',
        'tik_tok_latent_tokens', 'tik_tok_padding_classifier'
    )
    for attr in extra_attrs:
        if attr in compiled_names:
            continue
        if hasattr(vqvae, attr):
            child = getattr(vqvae, attr)
            if isinstance(child, nn.Module) and attr != 'vector_quantizer':
                setattr(vqvae, attr, torch.compile(child, mode=mode, backend=backend))

    return net


def compile_gcp_encoder(net: nn.Module, mode: str | None, backend: str | None) -> nn.Module:
    """Compile the GCP encoder component of ``SuperModel`` while leaving VQVAE untouched.

    Parameters
    ----------
    net : nn.Module
        The SuperModel instance.
    mode : str | None
        torch.compile optimization mode. Options: "reduce-overhead", "default", "max-autotune", or None.
        Use None to fall back to torch.compile's internal default behavior.
    backend : str | None
        torch.compile backend. Typical choices used here: "inductor" (recommended) or "eager".
        Use None to let torch decide / future default.

    Returns
    -------
    nn.Module
        The (in-place) modified model with its encoder compiled if present.
    """
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
