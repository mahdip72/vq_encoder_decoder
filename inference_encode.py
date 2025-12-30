import os
import yaml
import shutil
import datetime
import torch
import functools
from torch.utils.data import DataLoader
from box import Box
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list
import csv
from utils.utils import load_configs, load_checkpoints_simple, get_logging
from data.dataset import GCPNetDataset, custom_collate_pretrained_gcp
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
        cleaned = [int(v) for v in idx if v != -1]
        records.append({'pid': pid, 'indices': cleaned, 'protein_sequence': seq})


def main():
    # Load inference configuration
    with open("configs/inference_encode_config.yaml") as f:
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

    # Setup output directory with timestamp
    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    if accelerator.is_main_process:
        result_dir = os.path.join(infer_cfg.output_base_dir, now)
        os.makedirs(result_dir, exist_ok=True)
        shutil.copy("configs/inference_encode_config.yaml", result_dir)
        paths = [result_dir]
    else:
        # Initialize with placeholders.
        paths = [None]

    # Broadcast paths to all processes
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

    # Override task-specific settings
    configs.train_settings.max_task_samples = infer_cfg.get('max_task_samples', configs.train_settings.max_task_samples)
    configs.model.max_length = infer_cfg.get('max_length', configs.model.max_length)
    esm_cfg = getattr(configs.train_settings.losses, 'esm', None)
    if esm_cfg and getattr(esm_cfg, 'enabled', False):
        esm_cfg.enabled = False
    configs.model.encoder.pretrained.enabled = False

    # Load encoder/decoder configs from saved results instead of default utils
    encoder_configs, decoder_configs = load_saved_encoder_decoder_configs(
        encoder_cfg_path,
        decoder_cfg_path
    )

    # Prepare dataset and dataloader
    esm_tokenizer = None
    esm_cfg = getattr(configs.train_settings.losses, 'esm', None)
    if esm_cfg and getattr(esm_cfg, 'enabled', False):
        from transformers import AutoTokenizer
        esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    dataset = GCPNetDataset(
        infer_cfg.data_path,
        top_k=encoder_configs.top_k,
        num_positional_embeddings=encoder_configs.num_positional_embeddings,
        configs=configs,
        mode='evaluation',
        esm_tokenizer=esm_tokenizer,
    )
    collate_fn = functools.partial(
        custom_collate_pretrained_gcp,
        featuriser=dataset.pretrained_featuriser,
        task_transform=dataset.pretrained_task_transform,
    )

    loader = DataLoader(
        dataset,
        shuffle=infer_cfg.shuffle,
        batch_size=infer_cfg.batch_size,
        num_workers=infer_cfg.num_workers,
        collate_fn=collate_fn
    )

    # Setup file logger in result directory
    logger = get_logging(result_dir, configs)

    # Prepare model
    model = prepare_model(
        configs, logger,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs,
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
        drop_prefixes=["protein_encoder.", "vqvae.decoder.esm_"],
    )

    compile_cfg = infer_cfg.get('compile_model')
    if compile_cfg and compile_cfg.get('enabled', False):
        compile_mode = compile_cfg.get('mode')
        compile_backend = compile_cfg.get('backend', 'inductor')
        compile_encoder = compile_cfg.get('compile_encoder', True)

        if compile_encoder and hasattr(model, 'encoder') and getattr(configs.model.encoder, 'name', None) == 'gcpnet':
            model = compile_gcp_encoder(model, mode=compile_mode, backend=compile_backend)
            logger.info('GCP encoder compiled for inference.')

        model = compile_non_gcp_and_exclude_vq(model, mode=compile_mode, backend=compile_backend)
        logger.info('Compiled VQVAE components for inference (VQ layer excluded).')

    # Prepare everything with accelerator (model and dataloader)
    model, loader = accelerator.prepare(model, loader)

    # Prepare for optional VQ index recording
    indices_records = []  # list of dicts {'pid': str, 'indices': list[int]}

    # Initialize the progress bar using tqdm (separate from iteration)
    progress_bar = tqdm(range(0, int(len(loader))),
                        leave=True, disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process))
    progress_bar.set_description("Inference")

    for i, batch in enumerate(loader):
        # Inference loop
        with torch.inference_mode():
            # Move graph batch onto accelerator device
            batch['graph'] = batch['graph'].to(accelerator.device)
            batch['masks'] = batch['masks'].to(accelerator.device)
            batch['nan_masks'] = batch['nan_masks'].to(accelerator.device)

            # Forward pass: get either decoded outputs or VQ layer outputs
            output_dict = model(batch, return_vq_layer=True)
            indices = output_dict['indices']
            pids = batch['pid']  # list of identifiers
            sequences = batch['seq']

            # record indices per sample
            record_indices(pids, indices, sequences, indices_records)

            # Update progress bar manually
            progress_bar.update(1)

    # end progress_bar
    progress_bar.close()

    # Ensure all processes have completed before saving results
    accelerator.wait_for_everyone()

    # Gather outputs from all processes
    indices_records = accelerator.gather_for_metrics(indices_records, use_gather_object=True)

    if accelerator.is_main_process:
        # After loop, save indices CSV if requested
        csv_filename = infer_cfg.get('vq_indices_csv_filename', 'vq_indices.csv')
        csv_path = os.path.join(result_dir, csv_filename)
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

    logger.info(f"Inference encoding completed. Results are saved in {result_dir}")

    # Ensure all processes have completed before exiting
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()


if __name__ == '__main__':
    main()
