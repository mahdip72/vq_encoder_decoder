import os
import yaml
import shutil
import datetime
import torch
import functools
from torch.utils.data import DataLoader
from box import Box
from tqdm import tqdm
from accelerate import Accelerator
import csv

from utils.utils import load_configs, save_backbone_pdb_inference, load_checkpoints_simple, get_logging
from utils.custom_losses import calculate_aligned_mse_loss
from data.dataset import GCPNetDataset, custom_collate_pretrained_gcp, custom_collate
from models.super_model import prepare_model_vqvae


def load_saved_encoder_decoder_configs(encoder_cfg_path, decoder_cfg_path):
    # Load encoder and decoder configs from a saved result directory
    with open(encoder_cfg_path) as f:
        enc_cfg = yaml.full_load(f)
    encoder_configs = Box(enc_cfg)

    with open(decoder_cfg_path) as f:
        dec_cfg = yaml.full_load(f)
    decoder_configs = Box(dec_cfg)

    return encoder_configs, decoder_configs


def record_indices(pids, indices_tensor, sequences, records):
    """Append pid-index-sequence tuples to records list, ensuring indices is always a list."""
    cpu_inds = indices_tensor.detach().cpu().tolist()
    # Handle scalar to list
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    for pid, idx, seq in zip(pids, cpu_inds, sequences):
        # wrap non-list idx into list
        if not isinstance(idx, list):
            idx = [idx]
        records.append({'pid': pid, 'indices': idx[:len(seq)], 'protein_sequence': seq})


def save_predictions_to_pdb(pids, preds, masks, pdb_dir):
    """Save backbone PDB files for each sample in the batch."""
    for pid, coord, mask in zip(pids, preds, masks):
        prefix = os.path.join(pdb_dir, pid)
        save_backbone_pdb_inference(coord, mask, prefix)


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

    if infer_cfg.get('save_original_pdb', False):
        original_pdb_dir = os.path.join(result_dir, 'original_pdb_files')
        os.makedirs(original_pdb_dir, exist_ok=True)

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
        shuffle=infer_cfg['shuffle'],
        batch_size=infer_cfg['batch_size'],
        num_workers=infer_cfg['num_workers'],
        collate_fn=collate_fn
    )

    # Initialize accelerator for mixed precision and multi-GPU
    accelerator = Accelerator(mixed_precision=infer_cfg['mixed_precision'])
    # Setup file logger in result directory
    logger = get_logging(result_dir, configs)

    # Prepare model
    model = prepare_model_vqvae(
        configs, logger, accelerator,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs
    )
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(infer_cfg['trained_model_dir'], infer_cfg['checkpoint_path'])
    model = load_checkpoints_simple(checkpoint_path, model)

    # Prepare everything with accelerator (model and dataloader)
    model, list_loader = accelerator.prepare(model, [loader])
    loader = list_loader[0]  # Unpack the list since we only have one DataLoader

    # Prepare for optional VQ index recording
    indices_records = []  # list of dicts {'pid': str, 'indices': list[int]}


    # enable or disable progress bar
    iterator = (tqdm(loader, desc="Inference", total=len(loader))
                if infer_cfg.get('tqdm_progress_bar', True) else loader)
    for batch in iterator:
        # Inference loop
        with torch.inference_mode():
            # Move graph batch onto accelerator device
            batch['graph'] = batch['graph'].to(accelerator.device)

            # Forward pass: get either decoded outputs or VQ layer outputs
            output, indices, _ = model(batch, return_vq_layer=infer_cfg['return_vq_layer'])
            pids = batch['pid']  # list of identifiers
            if infer_cfg['return_vq_layer']:
                sequences = batch['seq']
                # record indices per sample
                record_indices(pids, indices, sequences, indices_records)

            else:
                # output is tuple of (bb_pred, ...)
                bb_pred = output[0]
                # reshape from (B, L, 9) to (B, L, 3, 3)
                preds = bb_pred.view(bb_pred.shape[0], bb_pred.shape[1], 3, 3)
                masks = batch['masks']
                true_coords = batch['target_coords'].view(preds.shape[0], preds.shape[1], 3, 3)

                # Align predicted coordinates to true coordinates
                _, preds_aligned, trues_aligned = calculate_aligned_mse_loss(
                    x_predicted=preds,
                    x_true=true_coords.to(accelerator.device),
                    masks=masks.to(accelerator.device),
                    alignment_strategy=infer_cfg.get('alignment_strategy', 'kabsch')
                )
                # save PDBs via helper
                save_predictions_to_pdb(pids, preds_aligned.detach().cpu(), masks.cpu(), pdb_dir)

                if infer_cfg.get('save_original_pdb', False):
                    # The ground truth coordinates are now aligned and can be saved
                    save_predictions_to_pdb(pids, trues_aligned.detach().cpu(), masks.cpu(), original_pdb_dir)

    logger.info(f"Inference completed. Results are saved in {result_dir}")
    # After loop, save indices CSV if requested
    if infer_cfg.get('return_vq_layer', False):
        csv_path = os.path.join(result_dir, 'vq_indices.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pid', 'indices', 'protein_sequence'])
            for rec in indices_records:
                pid = rec['pid']
                inds = rec['indices']
                seq = rec['protein_sequence']
                # ensure a list for joining
                if not isinstance(inds, (list, tuple)):
                    inds = [inds]
                writer.writerow([pid, ' '.join(map(str, inds)), seq])


if __name__ == '__main__':
    main()
