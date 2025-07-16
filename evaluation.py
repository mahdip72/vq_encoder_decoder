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
from models.super_model import prepare_model
from utils.evaluation.tmscore import TMscoring  # Import TM-score evaluation


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


def evaluate_structures(pdb_dir, original_pdb_dir, result_dir, logger):
    """Evaluate TM-score and RMSD between predicted and original structures."""
    logger.info("Starting TM-score and RMSD evaluation...")

    # Get all predicted PDB files
    pred_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]

    if not pred_files:
        logger.warning("No PDB files found for evaluation")
        return

    results = []
    failed_evaluations = []

    # Process each predicted structure
    for pred_file in tqdm(pred_files, desc="Evaluating structures"):
        pred_path = os.path.join(pdb_dir, pred_file)

        # Find corresponding original file
        # Remove any prefixes and use the base name
        base_name = pred_file
        original_path = os.path.join(original_pdb_dir, base_name)

        if not os.path.exists(original_path):
            logger.warning(f"Original file not found for {pred_file}")
            failed_evaluations.append(pred_file)
            continue

        try:
            # Create TMscoring instance for this pair of files
            tm_scorer = TMscoring(pred_path, original_path)

            # Optimize alignment and get TM-score and RMSD
            _, tm_score, rmsd = tm_scorer.optimise()

            results.append({
                'pdb_file': pred_file,
                'tm_score': tm_score,
                'rmsd': rmsd
            })

            # logger.info(f"Evaluated {pred_file}: TM-score={tm_score:.4f}, RMSD={rmsd:.4f}")

        except Exception as e:
            logger.error(f"Failed to evaluate {pred_file}: {str(e)}")
            failed_evaluations.append(pred_file)

    # Save results to CSV
    if results:
        csv_path = os.path.join(result_dir, 'detailed_scores.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pdb_file', 'tm_score', 'rmsd'])
            writer.writeheader()
            writer.writerows(results)

        # Calculate and log summary statistics
        tm_scores = [r['tm_score'] for r in results]
        rmsds = [r['rmsd'] for r in results]

        avg_tm_score = sum(tm_scores) / len(tm_scores)
        avg_rmsd = sum(rmsds) / len(rmsds)

        logger.info(f"Evaluation completed for {len(results)} structures")
        logger.info(f"Average TM-score: {avg_tm_score:.4f}")
        logger.info(f"Average RMSD: {avg_rmsd:.4f}")
        logger.info(f"Results saved to: {csv_path}")

        # Save summary statistics
        summary_path = os.path.join(result_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Structure Evaluation Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"Total structures evaluated: {len(results)}\n")
            f.write(f"Failed evaluations: {len(failed_evaluations)}\n")
            f.write(f"Average TM-score: {avg_tm_score:.4f}\n")
            f.write(f"Average RMSD: {avg_rmsd:.4f}\n")
            f.write(f"TM-score range: {min(tm_scores):.4f} - {max(tm_scores):.4f}\n")
            f.write(f"RMSD range: {min(rmsds):.4f} - {max(rmsds):.4f}\n")

            if failed_evaluations:
                f.write(f"\nFailed evaluations:\n")
                for failed in failed_evaluations:
                    f.write(f"  - {failed}\n")

    else:
        logger.error("No structures were successfully evaluated")


def main():
    # Load inference configuration
    with open("configs/evaluation_config.yaml") as f:
        infer_cfg = yaml.full_load(f)

    # Setup output directory with timestamp
    now = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
    result_dir = os.path.join(infer_cfg['output_base_dir'], now)
    os.makedirs(result_dir, exist_ok=True)
    pdb_dir = os.path.join(result_dir, 'pdb_files')
    os.makedirs(pdb_dir, exist_ok=True)

    original_pdb_dir = os.path.join(result_dir, 'original_pdb_files')
    os.makedirs(original_pdb_dir, exist_ok=True)

    # Copy evaluation config for reference
    shutil.copy("configs/evaluation_config.yaml", result_dir)

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
    model = prepare_model(
        configs, logger,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs
    )
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    # Load checkpoint
    checkpoint_path = os.path.join(infer_cfg['trained_model_dir'], infer_cfg['checkpoint_path'])
    model = load_checkpoints_simple(checkpoint_path, model, logger)

    # Prepare everything with accelerator (model and dataloader)
    model, list_loader = accelerator.prepare(model, [loader])
    loader = list_loader[0]  # Unpack the list since we only have one DataLoader

    # Prepare for optional VQ index recording
    indices_records = []  # list of dicts {'pid': str, 'indices': list[int]}


    # enable or disable progress bar
    iterator = (tqdm(loader, desc="Evaluation", total=len(loader), leave=True, disable=not (infer_cfg["tqdm_progress_bar"] and accelerator.is_main_process))
                if infer_cfg["tqdm_progress_bar"] else loader)
    for batch in iterator:
        # Evaluation loop
        with torch.inference_mode():
            # Move graph batch onto accelerator device
            batch['graph'] = batch['graph'].to(accelerator.device)
            batch['masks'] = batch['masks'].to(accelerator.device)
            batch['nan_masks'] = batch['nan_masks'].to(accelerator.device)

            # Forward pass: get either decoded outputs or VQ layer outputs
            output, indices, _ = model(batch)
            pids = batch['pid']  # list of identifiers
            sequences = batch['seq']
            # record indices per sample
            record_indices(pids, indices, sequences, indices_records)

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
            if accelerator.is_main_process:
                # save PDBs via helper
                save_predictions_to_pdb(pids, preds_aligned.detach().cpu(), masks.cpu(), pdb_dir)

                # The ground truth coordinates are now aligned and can be saved
                save_predictions_to_pdb(pids, trues_aligned.detach().cpu(), masks.cpu(), original_pdb_dir)


    logger.info(f"Evaluation completed. Results are saved in {result_dir}")

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

        # Evaluate structures using TM-score and RMSD
        evaluate_structures(pdb_dir, original_pdb_dir, result_dir, logger)


if __name__ == '__main__':
    main()
