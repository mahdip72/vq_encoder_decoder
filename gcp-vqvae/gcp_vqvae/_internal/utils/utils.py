import logging
from collections import OrderedDict
from typing import Iterable, Mapping, Optional

import h5py
import torch
from box import Box


def get_logger(name: str = "gcp_vqvae") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def get_nb_trainable_parameters(model, *, count_buffers: bool = False):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    buffer_params = 0
    if count_buffers:
        for _, buffer in model.named_buffers():
            buffer_params += buffer.numel()

    return trainable_params, all_param, buffer_params


def print_trainable_parameters(model, logging_obj, description="", *, include_buffers: bool = False):
    trainable_params, all_param, buffer_params = get_nb_trainable_parameters(
        model, count_buffers=include_buffers
    )
    total_params = all_param + (buffer_params if include_buffers else 0)

    if total_params == 0:
        logging_obj.info(
            "%s trainable params: %s || all params: %s || trainable%%: N/A (no parameters)",
            description,
            f"{trainable_params: ,}",
            f"{total_params: ,}",
        )
        return

    buffer_details = ""
    if include_buffers and buffer_params > 0:
        buffer_details = f" (buffers with EMA: {buffer_params: ,})"

    logging_obj.info(
        "%s trainable params: %s || all params: %s%s || trainable%%: %s",
        description,
        f"{trainable_params: ,}",
        f"{total_params: ,}",
        buffer_details,
        100 * trainable_params / total_params,
    )


def load_configs(config: Mapping) -> Box:
    tree_config = Box(config)
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config


def _remap_gcp_encoder_keys(state_dict: Mapping[str, torch.Tensor], model, logger=None):
    if not isinstance(state_dict, (dict, OrderedDict)):
        return state_dict

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return state_dict

    try:
        from gcp_vqvae._internal.models.gcpnet.models.base import PretrainedEncoder
    except ImportError:
        PretrainedEncoder = ()  # type: ignore

    def _is_unwrapped_key(key: str) -> bool:
        return (
            key.startswith("encoder.")
            and not key.startswith("encoder.encoder.")
            and not key.startswith("encoder.featuriser.")
            and not key.startswith("encoder.task_transform.")
        )

    has_wrapped_keys = any(key.startswith("encoder.encoder.") for key in state_dict)
    has_unwrapped_keys = any(_is_unwrapped_key(key) for key in state_dict)
    is_wrapped_model = isinstance(encoder, PretrainedEncoder)

    if not is_wrapped_model and has_wrapped_keys:
        remapped = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("encoder.encoder."):
                remapped["encoder." + key[len("encoder.encoder."):]] = value
            elif key.startswith("encoder.featuriser.") or key.startswith("encoder.task_transform."):
                continue
            else:
                remapped[key] = value
        if logger is not None:
            logger.info("Remapped encoder checkpoint keys for non-pretrained encoder.")
        return remapped

    if is_wrapped_model and not has_wrapped_keys and has_unwrapped_keys:
        remapped = OrderedDict()
        for key, value in state_dict.items():
            if _is_unwrapped_key(key):
                remapped["encoder.encoder." + key[len("encoder."):]] = value
            else:
                remapped[key] = value
        if logger is not None:
            logger.info("Remapped encoder checkpoint keys for pretrained encoder wrapper.")
        return remapped

    return state_dict


def load_checkpoints_simple(
    checkpoint_path: str,
    net,
    logger,
    *,
    decoder_only: bool = False,
    drop_prefixes: Optional[Iterable[str]] = None,
):
    model_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained_state_dict = model_checkpoint.get("model_state_dict", model_checkpoint)

    pretrained_state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in pretrained_state_dict.items()
    }

    pretrained_state_dict = _remap_gcp_encoder_keys(pretrained_state_dict, net, logger)

    if decoder_only:
        pretrained_state_dict = {
            k: v
            for k, v in pretrained_state_dict.items()
            if not (k.startswith("encoder") or k.startswith("vqvae.encoder"))
        }
    if drop_prefixes:
        prefixes = tuple(drop_prefixes)
        pretrained_state_dict = {
            k: v
            for k, v in pretrained_state_dict.items()
            if not any(k.startswith(prefix) for prefix in prefixes)
        }

    load_log = net.load_state_dict(pretrained_state_dict, strict=False)
    if logger is not None:
        logger.info("Loading checkpoint log: %s", load_log)
    return net


def load_h5_file(file_path: str):
    with h5py.File(file_path, "r") as handle:
        seq = handle["seq"][()]
        n_ca_c_o_coord = handle["N_CA_C_O_coord"][:]
        plddt_scores = handle["plddt_scores"][:]
    return seq, n_ca_c_o_coord, plddt_scores
