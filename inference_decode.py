import os
import yaml
import shutil
import datetime
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from box import Box
from tqdm import tqdm
from accelerate import Accelerator

from utils.utils import load_configs, save_backbone_pdb_inference, load_checkpoints_simple, get_logging
from models.super_model import prepare_model


class VQIndicesDataset(Dataset):
    """Dataset for loading VQ indices from a CSV file."""
    def __init__(self, csv_path, max_length):
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row['pid']
        # Indices are space-separated strings
        indices = [int(i) for i in row['indices'].split()]
        seq = row['protein_sequence']

        current_length = len(indices)
        pad_length = self.max_length - current_length

        # Pad indices with -1 and create a mask
        padded_indices = indices + [-1] * pad_length
        mask = [True] * current_length + [False] * pad_length

        return {
            'pid': pid,
            'indices': torch.tensor(padded_indices, dtype=torch.long),
            'seq': seq,
            'mask': torch.tensor(mask, dtype=torch.bool)
        }


def load_saved_decoder_config(decoder_cfg_path):
    # Load decoder config from a saved result directory
    with open(decoder_cfg_path) as f:
        dec_cfg = yaml.full_load(f)
    decoder_configs = Box(dec_cfg)
    return decoder_configs


def save_predictions_to_pdb(pids, preds, masks, pdb_dir):
    """Save backbone PDB files for each sample in the batch."""
    for pid, coord, mask in zip(pids, preds, masks):
        prefix = os.path.join(pdb_dir, pid)
        save_backbone_pdb_inference(coord, mask, prefix)


def main():
    # Load inference configuration
    with open("configs/inference_decode_config.yaml") as f:
        infer_cfg = yaml.full_load(f)

    # Setup output directory with timestamp
    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    result_dir = os.path.join(infer_cfg['output_base_dir'], now)
    os.makedirs(result_dir, exist_ok=True)
    pdb_dir = os.path.join(result_dir, 'pdb_files')
    os.makedirs(pdb_dir, exist_ok=True)

    # Copy inference config for reference
    shutil.copy("configs/inference_decode_config.yaml", result_dir)

    # Paths to training configs
    vqvae_cfg_path = os.path.join(infer_cfg["trained_model_dir"], infer_cfg['config_vqvae'])
    decoder_cfg_path = os.path.join(infer_cfg["trained_model_dir"], infer_cfg['config_decoder'])

    # Load main config
    with open(vqvae_cfg_path) as f:
        vqvae_cfg = yaml.full_load(f)
    configs = load_configs(vqvae_cfg)

    # Override task-specific settings
    configs.model.max_length = infer_cfg.get('max_length', configs.model.max_length)

    # Load decoder config from saved results
    decoder_configs = load_saved_decoder_config(decoder_cfg_path)

    # Prepare dataset and dataloader
    dataset = VQIndicesDataset(
        infer_cfg['indices_csv_path'],
        max_length=configs.model.max_length
    )

    loader = DataLoader(
        dataset,
        shuffle=infer_cfg['shuffle'],
        batch_size=infer_cfg['batch_size'],
        num_workers=infer_cfg['num_workers']
    )

    # Initialize accelerator for mixed precision and multi-GPU
    accelerator = Accelerator(mixed_precision=infer_cfg['mixed_precision'])
    # Setup file logger in result directory
    logger = get_logging(result_dir, configs)

    # Prepare model (decoder only)
    model = prepare_model(
        configs, logger,
        decoder_configs=decoder_configs,
        decoder_only=True
    )
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(infer_cfg['trained_model_dir'], infer_cfg['checkpoint_path'])
    model = load_checkpoints_simple(checkpoint_path, model, logger, decoder_only=True)

    # Prepare everything with accelerator (model and dataloader)
    model, loader = accelerator.prepare(model, loader)

    # enable or disable progress bar
    iterator = (tqdm(loader, desc="Inference", total=len(loader))
                if infer_cfg.get('tqdm_progress_bar', True) else loader)
    for batch in iterator:
        # Inference loop
        with torch.inference_mode():
            indices = batch['indices']
            masks = batch['mask']

            # Forward pass through the decoder
            output, _, _ = model(batch, decoder_only=True)
            
            bb_pred = output[0]
            preds = bb_pred.view(bb_pred.shape[0], bb_pred.shape[1], 3, 3)
            
            pids = batch['pid']
            
            save_predictions_to_pdb(pids, preds.detach().cpu(), masks.cpu(), pdb_dir)

    logger.info(f"Inference completed. Results are saved in {result_dir}")


if __name__ == '__main__':
    main()
