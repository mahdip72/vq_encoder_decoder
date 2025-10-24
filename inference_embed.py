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
import h5py
import numpy as np

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


def record_embeddings(pids, embeddings_array, indices_tensor, sequences, records):
    """Append pid-embedding-indices-sequence tuples to records list."""
    # embeddings_array: numpy array (B, L, D)
    cpu_inds = indices_tensor.detach().cpu().tolist()
    for pid, emb, ind_list, seq in zip(pids, embeddings_array, cpu_inds, sequences):
        # Trim to sequence length
        L = len(seq)
        emb_trim = emb[:L].astype('float32')
        # cleaned = [int(v) for v in ind_list if v != -1]
        records.append({'pid': pid, 'embedding': emb_trim, 'indices': ind_list[:L], 'protein_sequence': seq})


def main():
    # Load inference configuration
    with open("configs/inference_embed_config.yaml") as f:
        infer_cfg = yaml.full_load(f)
    infer_cfg = Box(infer_cfg)

    dataloader_config = DataLoaderConfiguration(
        non_blocking=True,
        even_batches=False
    )

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=infer_cfg.mixed_precision,
        dataloader_config=dataloader_config
    )

    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result_dir = os.path.join(infer_cfg.output_base_dir, now)
        os.makedirs(result_dir, exist_ok=True)
        shutil.copy("configs/inference_embed_config.yaml", result_dir)
        paths = [result_dir]
    else:
        paths = [None]

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

    # Load encoder/decoder configs from saved results
    encoder_configs, decoder_configs = load_saved_encoder_decoder_configs(
        encoder_cfg_path,
        decoder_cfg_path
    )

    # Prepare dataset and dataloader
    dataset = GCPNetDataset(
        infer_cfg.data_path,
        top_k=encoder_configs.top_k,
        num_positional_embeddings=encoder_configs.num_positional_embeddings,
        configs=configs,
        mode='evaluation'
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

    # Setup file logger
    logger = get_logging(result_dir, configs)

    # Prepare model
    model = prepare_model(
        configs, logger,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs
    )
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    checkpoint_path = os.path.join(infer_cfg.trained_model_dir, infer_cfg.checkpoint_path)
    model = load_checkpoints_simple(checkpoint_path, model, logger)

    compile_cfg = infer_cfg.get('compile_model')
    if compile_cfg and compile_cfg.get('enabled', False):
        compile_mode = compile_cfg.get('mode')
        compile_backend = compile_cfg.get('backend', 'inductor')
        compile_encoder = compile_cfg.get('compile_encoder', True)

        if compile_encoder and hasattr(model, 'encoder') and getattr(configs.model.encoder, 'name', None) == 'gcpnet':
            model = compile_gcp_encoder(model, mode=compile_mode, backend=compile_backend)
            logger.info('GCP encoder compiled for embedding inference.')

        model = compile_non_gcp_and_exclude_vq(model, mode=compile_mode, backend=compile_backend)
        logger.info('Compiled VQVAE components for embedding inference (VQ layer excluded).')

    model, loader = accelerator.prepare(model, loader)

    embeddings_records = []  # list of dicts {'pid', 'embedding', 'protein_sequence'}

    progress_bar = tqdm(range(0, int(len(loader))), leave=True,
                        disable=not (infer_cfg.tqdm_progress_bar and accelerator.is_main_process))
    progress_bar.set_description("Inference embed")

    for i, batch in enumerate(loader):
        with torch.inference_mode():
            # move graph batch to device
            if 'graph' in batch:
                batch['graph'] = batch['graph'].to(accelerator.device)
            batch['masks'] = batch['masks'].to(accelerator.device)
            batch['nan_masks'] = batch['nan_masks'].to(accelerator.device)

            # Forward pass to get embeddings from VQ layer
            output_dict = model(batch, return_vq_layer=True)
            embeddings = output_dict['embeddings']
            indices = output_dict['indices']
            pids = batch['pid']
            sequences = batch['seq']

            emb_np = embeddings.detach().cpu().numpy()
            record_embeddings(pids, emb_np, indices, sequences, embeddings_records)

            progress_bar.update(1)

    # end progress_bar
    progress_bar.close()

    accelerator.wait_for_everyone()

    # Gather records from all processes
    embeddings_records = accelerator.gather_for_metrics(embeddings_records, use_gather_object=True)

    if accelerator.is_main_process:
        h5_path = os.path.join(result_dir, infer_cfg.vq_embeddings_h5_filename)
        with h5py.File(h5_path, 'w') as hf:
            for rec in embeddings_records:
                pid = rec['pid']
                emb = rec['embedding']
                inds = rec['indices']
                # create group per pid
                group = hf.create_group(pid)
                group.create_dataset('embedding', data=emb, compression='gzip')
                group.create_dataset('indices', data=np.array(inds, dtype=np.int32), compression='gzip')
        logger.info(f"Saved embeddings HDF5: {h5_path}")

    logger.info(f"Embedding extraction completed. Saving to HDF5 in {result_dir}")

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()

if __name__ == '__main__':
    main()
