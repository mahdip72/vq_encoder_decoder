import os
import yaml
import shutil
import datetime
import argparse
import torch
import h5py
from box import Box

from utils.utils import load_configs, load_checkpoints_simple, get_logging
from models.super_model import prepare_model


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
    """Try to locate the codebook tensor inside the vector-quantizer by inspecting
    its state_dict and parameters. Returns (tensor, key_name) or (None, None).
    """
    # Prefer state_dict tensors (captures buffers/params)
    for key, tensor in vq_layer.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 2:
            if expected_count is not None and expected_dim is not None:
                if tensor.shape[0] == expected_count and tensor.shape[1] == expected_dim:
                    return tensor, key
            else:
                # fallback: take the first 2D tensor
                return tensor, key

    # Fallback to named_parameters
    for key, param in vq_layer.named_parameters():
        if param.ndim == 2:
            if expected_count is not None and expected_dim is not None:
                if param.shape[0] == expected_count and param.shape[1] == expected_dim:
                    return param.data, key
            else:
                return param.data, key

    return None, None


def main(config_path: str):
    # Load inference configuration
    with open(config_path) as f:
        infer_cfg = yaml.full_load(f)
    infer_cfg = Box(infer_cfg)

    # Create result directory with timestamp
    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    result_dir = os.path.join(infer_cfg.output_base_dir, now)
    os.makedirs(result_dir, exist_ok=True)
    shutil.copy(config_path, result_dir)

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

    # Save to HDF5: store each code vector as dataset named by its index (string)
    h5_filename = infer_cfg['codebook_h5_filename']
    h5_path = os.path.join(result_dir, h5_filename)

    with h5py.File(h5_path, 'w') as hf:
        for idx, vec in enumerate(codebook_np):
            ds_name = str(idx)
            hf.create_dataset(ds_name, data=vec, compression='gzip')

    logger.info(f'Saved codebook embeddings ({codebook_np.shape[0]} x {codebook_np.shape[1]}) to {h5_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract VQ codebook embeddings and save to HDF5')
    parser.add_argument('--config_path', '-c', help='Path to inference_codebook_extraction_config.yaml',
                        default='./configs/inference_codebook_extraction_config.yaml')
    args = parser.parse_args()
    main(args.config_path)
