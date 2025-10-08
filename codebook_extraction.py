import os
import yaml
import shutil
import datetime
import argparse
import torch
import h5py
import numpy as np
from box import Box
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list

from utils.utils import load_configs, load_checkpoints_simple, get_logging
from models.super_model import (
    prepare_model,
    compile_non_gcp_and_exclude_vq,
    compile_gcp_encoder,
)


def load_saved_encoder_decoder_configs(encoder_cfg_path, decoder_cfg_path):
    # Load encoder and decoder configs from a saved result directory
    with open(encoder_cfg_path) as f:
        enc_cfg = yaml.full_load(f)
    encoder_configs = Box(enc_cfg)

    with open(decoder_cfg_path) as f:
        dec_cfg = yaml.full_load(f)
    decoder_configs = Box(dec_cfg)

    return encoder_configs, decoder_configs


def find_codebook_tensor(vq_layer, expected_count=None, expected_dim=None):
    """Return the `codebook` tensor from the vq_layer if available.
    This simplified function assumes the vector-quantizer exposes a `codebook`
    attribute (torch.Tensor) of shape (codebook_size, dim), and returns it
    directly. If expected_count/expected_dim are provided, the shape is
    validated before returning.
    """
    if hasattr(vq_layer, 'codebook'):
        cb = getattr(vq_layer, 'codebook')
        if isinstance(cb, torch.Tensor) and cb.ndim == 2:
            if expected_count is not None and expected_dim is not None:
                if cb.shape[0] == expected_count and cb.shape[1] == expected_dim:
                    return cb, 'codebook'
                else:
                    return None, None
            return cb, 'codebook'
    return None, None


def main(config_path: str):
    # Load inference configuration
    with open(config_path) as f:
        infer_cfg = yaml.full_load(f)
    infer_cfg = Box(infer_cfg)

    dataloader_config = DataLoaderConfiguration(
        non_blocking=True,
        even_batches=False
    )

    # Initialize accelerator (needed for accelerate logging utilities)
    accelerator = Accelerator(
        mixed_precision=infer_cfg.get('mixed_precision', None),
        dataloader_config=dataloader_config
    )

    # Create result directory with timestamp only on main process and broadcast
    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result_dir = os.path.join(infer_cfg.output_base_dir, now)
        os.makedirs(result_dir, exist_ok=True)
        shutil.copy(config_path, result_dir)
        paths = [result_dir]
    else:
        paths = [None]

    # Broadcast the result_dir to all processes
    broadcast_object_list(paths, from_process=0)
    result_dir = paths[0]

    # Paths to training configs
    vqvae_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_vqvae)
    encoder_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_encoder)
    decoder_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_decoder)

    # Load main config
    with open(vqvae_cfg_path) as f:
        vqvae_cfg = yaml.full_load(f)
    configs = load_configs(vqvae_cfg)

    # Load encoder/decoder configs from saved results
    encoder_configs, decoder_configs = load_saved_encoder_decoder_configs(
        encoder_cfg_path,
        decoder_cfg_path
    )

    # Setup logger
    logger = get_logging(result_dir, configs)

    # Prepare model
    model = prepare_model(
        configs, logger,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
    )

    # Freeze parameters and set eval
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.checkpoint_path)
    model = load_checkpoints_simple(checkpoint_path, model, logger)

    compile_cfg = infer_cfg.get('compile_model')
    if compile_cfg and compile_cfg.get('enabled', False):
        compile_mode = compile_cfg.get('mode')
        compile_backend = compile_cfg.get('backend', 'inductor')
        compile_encoder = compile_cfg.get('compile_encoder', True)

        if compile_encoder and hasattr(model, 'encoder') and getattr(configs.model.encoder, 'name', None) == 'gcpnet':
            model = compile_gcp_encoder(model, mode=compile_mode, backend=compile_backend)
            logger.info('GCP encoder compiled for codebook extraction.')

        model = compile_non_gcp_and_exclude_vq(model, mode=compile_mode, backend=compile_backend)
        logger.info('Compiled VQVAE components for codebook extraction (VQ layer excluded).')

    # Access vq layer
    if hasattr(model, 'vqvae') and hasattr(model.vqvae, 'vector_quantizer'):
        vq_layer = model.vqvae.vector_quantizer
    else:
        logger.error('Vector quantizer layer not found on the model (expected model.vqvae.vector_quantizer)')
        raise RuntimeError('Vector quantizer layer not found')

    expected_count = getattr(model.vqvae, 'codebook_size', None)
    expected_dim = getattr(model.vqvae, 'vqvae_dim', None)

    codebook_tensor, key_name = find_codebook_tensor(vq_layer, expected_count, expected_dim)

    if codebook_tensor is None:
        logger.error('Failed to locate a 2D codebook tensor in the vector-quantizer state.\n'
                     'Inspect the vq_layer.state_dict() keys to find the embedding tensor manually.')
        # Dump available keys for debugging
        keys = [k for k, v in vq_layer.state_dict().items()]
        logger.error(f'Available state_dict keys: {keys}')
        raise RuntimeError('Codebook tensor not found')

    # Ensure we have CPU numpy array
    codebook_np = codebook_tensor.detach().cpu().numpy().astype('float32')

    # Save to HDF5: store the entire codebook as a single dataset named 'codebook'
    h5_filename = infer_cfg.get('vq_embeddings_h5_filename', infer_cfg.get('codebook_h5_filename', 'codebook_embeddings.h5'))
    h5_path = os.path.join(result_dir, h5_filename)

    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('codebook', data=codebook_np, compression='gzip')
        indices = np.arange(codebook_np.shape[0], dtype=np.int32)
        hf.create_dataset('indices', data=indices, compression='gzip')

    logger.info(f"Saved codebook embeddings ({codebook_np.shape[0]} x {codebook_np.shape[1]}) and indices to {h5_path}")
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract VQ codebook embeddings and save to HDF5')
    parser.add_argument('--config_path', '-c', help='Path to inference_codebook_extraction_config.yaml',
                        default='./configs/inference_codebook_extraction_config.yaml')
    args = parser.parse_args()
    main(args.config_path)
