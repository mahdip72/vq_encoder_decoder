import os
import yaml
import shutil
import datetime
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from box import Box
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list

from utils.utils import load_configs, save_backbone_pdb_inference, load_checkpoints_simple, get_logging
from models.super_model import (
    prepare_model,
    compile_non_gcp_and_exclude_vq,
    compile_gcp_encoder,
)


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

        nan_mask = torch.tensor(mask, dtype=torch.bool)

        for i, value in enumerate(padded_indices):
            if value == -1:
                nan_mask[i] = False

        return {
            'pid': pid,
            'indices': torch.tensor(padded_indices, dtype=torch.long),
            'seq': seq,
            'masks': torch.tensor(mask, dtype=torch.bool),
            'nan_masks': nan_mask
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
    infer_cfg = Box(infer_cfg)

    dataloader_config = DataLoaderConfiguration(
        # dispatch_batches=False,
        non_blocking=True,
        even_batches=False
    )

    # Initialize accelerator for mixed precision and multi-GPU
    accelerator = Accelerator(
        mixed_precision=infer_cfg.mixed_precision,
        dataloader_config=dataloader_config
    )

    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Setup output directory with timestamp
        result_dir = os.path.join(infer_cfg.output_base_dir, now)
        os.makedirs(result_dir, exist_ok=True)
        pdb_dir = os.path.join(result_dir, 'pdb_files')
        os.makedirs(pdb_dir, exist_ok=True)

        # Copy inference config for reference
        shutil.copy("configs/inference_decode_config.yaml", result_dir)
        paths = [result_dir, pdb_dir]
    else:
        # Initialize with placeholders.
        paths = [None, None]

    # Broadcast paths to all processes
    broadcast_object_list(paths, from_process=0)
    result_dir, pdb_dir = paths

    # Paths to training configs
    vqvae_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_vqvae)
    decoder_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_decoder)

    # Load main config
    with open(vqvae_cfg_path) as f:
        vqvae_cfg = yaml.full_load(f)
    configs = load_configs(vqvae_cfg)

    # Override task-specific settings
    configs.model.max_length = infer_cfg.get('max_length', configs.model.max_length)
    esm_cfg = getattr(configs.train_settings.losses, 'esm', None)
    if esm_cfg and getattr(esm_cfg, 'enabled', False):
        esm_cfg.enabled = False
    configs.model.encoder.pretrained.enabled = False

    # Load decoder config from saved results
    decoder_configs = load_saved_decoder_config(decoder_cfg_path)

    # Prepare dataset and dataloader
    dataset = VQIndicesDataset(
        infer_cfg.indices_csv_path,
        max_length=configs.model.max_length
    )

    loader = DataLoader(
        dataset,
        shuffle=infer_cfg.shuffle,
        batch_size=infer_cfg.batch_size,
        num_workers=infer_cfg.num_workers
    )

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
    checkpoint_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.checkpoint_path)
    model = load_checkpoints_simple(
        checkpoint_path,
        model,
        logger,
        decoder_only=True,
        drop_prefixes=["protein_encoder.", "vqvae.decoder.esm_"],
    )

    compile_cfg = infer_cfg.get('compile_model')
    if compile_cfg and compile_cfg.get('enabled', False):
        compile_mode = compile_cfg.get('mode')
        compile_backend = compile_cfg.get('backend', 'inductor')
        compile_encoder = compile_cfg.get('compile_encoder', False)

        if compile_encoder and hasattr(model, 'encoder') and getattr(configs.model.encoder, 'name', None) == 'gcpnet':
            model = compile_gcp_encoder(model, mode=compile_mode, backend=compile_backend)
            logger.info('GCP encoder compiled for decode inference.')

        model = compile_non_gcp_and_exclude_vq(model, mode=compile_mode, backend=compile_backend)
        logger.info('Compiled decoder path for inference (VQ layer excluded).')

    # Prepare everything with accelerator (model and dataloader)
    model, loader = accelerator.prepare(model, loader)

    # Initialize the progress bar using tqdm (separate from iteration)
    progress_bar = tqdm(range(0, int(len(loader))),
                        leave=True, disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process))
    progress_bar.set_description("Inference")

    for i, batch in enumerate(loader):
        # Inference loop
        with torch.inference_mode():
            indices = batch['indices']
            masks = batch['masks']

            # Forward pass through the decoder
            output_dict = model(batch, decoder_only=True)

            bb_pred = output_dict["outputs"]
            preds = bb_pred.view(bb_pred.shape[0], bb_pred.shape[1], 3, 3)

            pids = batch['pid']

            save_predictions_to_pdb(pids, preds.detach().cpu(), masks.cpu(), pdb_dir)

            # Update progress bar manually
            progress_bar.update(1)

    # end progress_bar
    progress_bar.close()

    logger.info(f"Inference decoding completed. Results are saved in {result_dir}")

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()


if __name__ == '__main__':
    main()
