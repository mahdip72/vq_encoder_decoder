import os
import yaml
import shutil
import datetime
import torch
import logging
import functools
from torch.utils.data import DataLoader
from box import Box  # add Box for config loading
from tqdm import tqdm

from utils.utils import load_configs, save_backbone_pdb, load_checkpoints_simple
from data.dataset import GCPNetDataset, custom_collate_pretrained_gcp, custom_collate
from models.super_model import prepare_model_vqvae

# Dummy accelerator for compatibility
class DummyAccelerator:
    def __init__(self):
        self.is_main_process = True


def load_saved_encoder_decoder_configs(encoder_cfg_path, decoder_cfg_path):
    # Load encoder and decoder configs from a saved result directory
    with open(encoder_cfg_path) as f:
        enc_cfg = yaml.full_load(f)
    encoder_configs = Box(enc_cfg)

    with open(decoder_cfg_path) as f:
        dec_cfg = yaml.full_load(f)
    decoder_configs = Box(dec_cfg)

    return encoder_configs, decoder_configs


def main():
    # Load inference configuration
    with open("configs/inference_config.yaml") as f:
        infer_cfg = yaml.full_load(f)

    # Setup output directory with timestamp
    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    result_dir = os.path.join(infer_cfg['output_base_dir'], now)
    os.makedirs(result_dir, exist_ok=True)
    pdb_dir = os.path.join(result_dir, 'pdb_files')
    os.makedirs(pdb_dir, exist_ok=True)

    # Copy inference config for reference
    shutil.copy("configs/inference_config.yaml", result_dir)

    # Paths to training configs
    vqvae_cfg_path = os.path.join(infer_cfg["trained_model_dir"], infer_cfg['config_vqvae'])
    encoder_cfg_path = os.path.join(infer_cfg["trained_model_dir"], infer_cfg['config_encoder'])
    decoder_cfg_path = os.path.join(infer_cfg["trained_model_dir"], infer_cfg['config_decoder'])

    # Load main config
    with open(vqvae_cfg_path) as f:
        vqvae_cfg = yaml.full_load(f)
    configs = load_configs(vqvae_cfg)

    # Override task-specific settings
    configs.train_settings.max_task_samples = infer_cfg.get('max_task_samples', configs.train_settings.max_task_samples)
    configs.model.max_length = infer_cfg.get('max_length', configs.model.max_length)

    # Load encoder/decoder configs from saved results instead of default utils
    encoder_configs, decoder_configs = load_saved_encoder_decoder_configs(
        encoder_cfg_path,
        decoder_cfg_path
    )

    # Prepare dataset and dataloader
    dataset = GCPNetDataset(
        infer_cfg['data_path'],
        top_k=encoder_configs.top_k,
        num_positional_embeddings=encoder_configs.num_positional_embeddings,
        configs=configs,
        mode='evaluation'
    )
    # Select collate function
    if configs.model.encoder.pretrained.enabled:
        collate_fn = functools.partial(
            custom_collate_pretrained_gcp,
            featuriser=dataset.pretrained_featuriser,
            task_transform=dataset.pretrained_task_transform
        )
    else:
        collate_fn = custom_collate

    loader = DataLoader(
        dataset,
        batch_size=infer_cfg['batch_size'],
        num_workers=infer_cfg['num_workers'],
        collate_fn=collate_fn
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare model
    logger = logging.getLogger('inference')
    logger.setLevel(logging.INFO)
    model = prepare_model_vqvae(
        configs, logger, DummyAccelerator(),
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs
    )
    model.to(device)
    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(infer_cfg['trained_model_dir'], infer_cfg['checkpoint_path'])
    model = load_checkpoints_simple(checkpoint_path, model)
    model.to(device)

    # Inference loop
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", total=len(loader)):
            # Move batch elements to device
            batch['graph'] = batch['graph'].to(device)
            # Forward pass and unpack tuples
            net_outputs, _, _ = model(batch)
            # net_outputs is a tuple: (bb_pred (B, L, 9), dir_logits, dist_logits, seq_logits) or (x, None, None, None)
            bb_pred = net_outputs[0]
            # reshape from (B, L, 9) to (B, L, 3, 3)
            preds = bb_pred.view(bb_pred.shape[0], bb_pred.shape[1], 3, 3).detach().cpu()
            masks = batch['masks'].cpu()
            pids = batch['pid']  # list of identifiers

            for i, pid in enumerate(pids):
                coords = preds[i]
                mask = masks[i]
                prefix = os.path.join(pdb_dir, pid)
                save_backbone_pdb(coords, mask, prefix)

    logger.info(f"Inference completed. Results are saved in {result_dir}")


if __name__ == '__main__':
    main()
