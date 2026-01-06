import argparse
import datetime
import os
import shutil
import sys
from contextlib import nullcontext

import torch
import yaml
from box import Box
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if not os.path.exists(os.path.join(PROJECT_ROOT, "configs")):
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.custom_losses import calculate_aligned_mse_loss
from models.super_model import prepare_model
from data.dataset import custom_collate_pretrained_gcp

from dataset import DemoStructureDataset
from demo_utils import (
    build_logger,
    load_saved_encoder_decoder_configs,
    load_configs,
    load_checkpoints_simple,
    record_indices,
    record_embeddings,
    save_predictions_to_pdb,
    write_indices_csv,
    write_embeddings_h5,
    evaluate_structures,
)


def parse_args():
    default_config = os.path.join(os.path.dirname(__file__), "demo_eval_config.yaml")
    parser = argparse.ArgumentParser(description="Lightweight demo evaluation (PDB/CIF -> tokens -> reconstruction)")
    parser.add_argument("--config", default=default_config, help="Path to demo evaluation config YAML")
    return parser.parse_args()


def resolve_encoder_config_path(infer_cfg):
    trained_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_encoder)
    if os.path.exists(trained_path):
        return trained_path
    return os.path.join("configs", "config_gcpnet_encoder.yaml")


def autocast_context(mixed_precision, device):
    if device.type != "cuda":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    if mixed_precision == "fp16":
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def main():
    args = parse_args()

    with open(args.config) as f:
        infer_cfg = Box(yaml.full_load(f))

    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    result_dir = os.path.join(infer_cfg.output_base_dir, now)
    os.makedirs(result_dir, exist_ok=True)
    shutil.copy(args.config, result_dir)

    pdb_dir = os.path.join(result_dir, 'pdb_files')
    original_pdb_dir = os.path.join(result_dir, 'original_pdb_files')
    if infer_cfg.get('save_pdb_and_evaluate', True):
        os.makedirs(pdb_dir, exist_ok=True)
        os.makedirs(original_pdb_dir, exist_ok=True)

    vqvae_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_vqvae)
    encoder_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_encoder)
    decoder_cfg_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.config_decoder)

    with open(vqvae_cfg_path) as f:
        vqvae_cfg = yaml.full_load(f)
    configs = load_configs(vqvae_cfg)

    if infer_cfg.get('max_task_samples', 0):
        configs.train_settings.max_task_samples = infer_cfg.max_task_samples
    esm_cfg = getattr(configs.train_settings.losses, 'esm', None)
    if esm_cfg and getattr(esm_cfg, 'enabled', False):
        esm_cfg.enabled = False
    configs.model.encoder.pretrained.enabled = False

    encoder_configs, decoder_configs = load_saved_encoder_decoder_configs(
        encoder_cfg_path,
        decoder_cfg_path,
    )

    encoder_config_path = resolve_encoder_config_path(infer_cfg)

    dataset = DemoStructureDataset(
        infer_cfg.data_dir,
        max_length=configs.model.max_length,
        encoder_config_path=encoder_config_path,
        max_task_samples=infer_cfg.get('max_task_samples', 0),
        progress=infer_cfg.get('tqdm_progress_bar', True),
    )

    logger = build_logger(result_dir)
    logger.info(f"Processed {len(dataset)} samples")
    if dataset.stats:
        logger.info(f"Processing stats: {dict(dataset.stats)}")

    collate_fn = lambda batch: custom_collate_pretrained_gcp(
        batch,
        featuriser=dataset.pretrained_featuriser,
        task_transform=dataset.pretrained_task_transform,
    )

    loader = DataLoader(
        dataset,
        shuffle=infer_cfg.get('shuffle', False),
        batch_size=infer_cfg.get('batch_size', 1),
        num_workers=infer_cfg.get('num_workers', 0),
        collate_fn=collate_fn,
    )

    model = prepare_model(
        configs,
        logger,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    checkpoint_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.checkpoint_path)
    model = load_checkpoints_simple(
        checkpoint_path,
        model,
        logger,
        drop_prefixes=["protein_encoder.", "vqvae.decoder.esm_"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    indices_records = []
    embeddings_records = []

    iterator = tqdm(
        loader,
        total=len(loader),
        desc="Inference",
        disable=not infer_cfg.get('tqdm_progress_bar', True),
    )
    for batch in iterator:
        batch['graph'] = batch['graph'].to(device)
        batch['masks'] = batch['masks'].to(device)
        batch['nan_masks'] = batch['nan_masks'].to(device)
        batch['target_coords'] = batch['target_coords'].to(device)

        with torch.inference_mode(), autocast_context(infer_cfg.get('mixed_precision', 'no'), device):
            output_dict = model(batch)

        indices = output_dict['indices']
        pids = batch['pid']
        sequences = batch['seq']

        if infer_cfg.get('save_indices_csv', False):
            record_indices(
                pids,
                indices,
                sequences,
                indices_records,
                max_length=configs.model.max_length,
            )

        bb_pred = output_dict["outputs"]
        preds = bb_pred.view(bb_pred.shape[0], bb_pred.shape[1], 3, 3)
        masks = torch.logical_and(batch['masks'], batch['nan_masks'])
        true_coords = batch['target_coords'].view(preds.shape[0], preds.shape[1], 3, 3)

        _, preds_aligned, trues_aligned = calculate_aligned_mse_loss(
            x_predicted=preds,
            x_true=true_coords,
            masks=masks,
            alignment_strategy=infer_cfg.get('alignment_strategy', 'kabsch'),
        )

        if infer_cfg.get('save_pdb_and_evaluate', True):
            save_predictions_to_pdb(pids, preds_aligned.detach().cpu(), masks.cpu(), pdb_dir)
            save_predictions_to_pdb(pids, trues_aligned.detach().cpu(), masks.cpu(), original_pdb_dir)

        if infer_cfg.get('save_embeddings_h5', False):
            with torch.inference_mode(), autocast_context(infer_cfg.get('mixed_precision', 'no'), device):
                vq_dict = model(batch, return_vq_layer=True)
            embeddings = vq_dict['embeddings']
            vq_indices = vq_dict['indices']
            emb_np = embeddings.detach().cpu().numpy()
            record_embeddings(
                pids,
                emb_np,
                vq_indices,
                sequences,
                embeddings_records,
                max_length=configs.model.max_length,
            )

    if infer_cfg.get('save_indices_csv', False):
        csv_path = os.path.join(result_dir, infer_cfg.get('indices_csv_filename', 'vq_indices.csv'))
        write_indices_csv(csv_path, indices_records)
        logger.info(f"Saved indices CSV: {csv_path}")

    if infer_cfg.get('save_embeddings_h5', False):
        h5_path = os.path.join(result_dir, infer_cfg.get('embeddings_h5_filename', 'vq_embed.h5'))
        write_embeddings_h5(h5_path, embeddings_records)
        logger.info(f"Saved embeddings HDF5: {h5_path}")

    if infer_cfg.get('save_pdb_and_evaluate', True):
        evaluate_structures(
            pdb_dir,
            original_pdb_dir,
            result_dir,
            logger,
            show_progress=infer_cfg.get('tqdm_progress_bar', True),
        )

    logger.info(f"Demo evaluation completed. Results are saved in {result_dir}")


if __name__ == '__main__':
    main()
