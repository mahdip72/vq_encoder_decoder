import csv
import logging
import os
import sys
from collections import OrderedDict

import h5py
import numpy as np
import torch
import yaml
from box import Box
from tqdm import tqdm

from utils.evaluation.tmscore import TMscoring


def load_saved_encoder_decoder_configs(encoder_cfg_path, decoder_cfg_path):
    with open(encoder_cfg_path) as f:
        enc_cfg = yaml.full_load(f)
    encoder_configs = Box(enc_cfg)

    with open(decoder_cfg_path) as f:
        dec_cfg = yaml.full_load(f)
    decoder_configs = Box(dec_cfg)

    return encoder_configs, decoder_configs


def load_configs(config):
    tree_config = Box(config)
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    return tree_config


def _remap_gcp_encoder_keys(state_dict, model, logger=None):
    if not isinstance(state_dict, (dict, OrderedDict)):
        return state_dict

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return state_dict

    try:
        from models.gcpnet.models.base import PretrainedEncoder
    except ImportError:
        PretrainedEncoder = ()  # type: ignore

    def _is_unwrapped_key(key: str) -> bool:
        return (
            key.startswith("encoder.")
            and not key.startswith("encoder.encoder.")
            and not key.startswith("encoder.featuriser.")
            and not key.startswith("encoder.task_transform.")
        )

    has_wrapped_keys = any(key.startswith("encoder.encoder.") for key in state_dict)
    has_unwrapped_keys = any(_is_unwrapped_key(key) for key in state_dict)
    is_wrapped_model = isinstance(encoder, PretrainedEncoder)

    if not is_wrapped_model and has_wrapped_keys:
        remapped = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("encoder.encoder."):
                remapped["encoder." + key[len("encoder.encoder."):]] = value
            elif key.startswith("encoder.featuriser.") or key.startswith("encoder.task_transform."):
                continue
            else:
                remapped[key] = value
        if logger is not None:
            logger.info("Remapped encoder checkpoint keys for non-pretrained encoder.")
        return remapped

    if is_wrapped_model and not has_wrapped_keys and has_unwrapped_keys:
        remapped = OrderedDict()
        for key, value in state_dict.items():
            if _is_unwrapped_key(key):
                remapped["encoder.encoder." + key[len("encoder."):]] = value
            else:
                remapped[key] = value
        if logger is not None:
            logger.info("Remapped encoder checkpoint keys for pretrained encoder wrapper.")
        return remapped

    return state_dict


def load_checkpoints_simple(checkpoint_path, net, logger, decoder_only=False, drop_prefixes=None):
    model_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    pretrained_state_dict = model_checkpoint['model_state_dict']

    pretrained_state_dict = {
        k.replace('_orig_mod.', ''): v for k, v in pretrained_state_dict.items()
    }

    pretrained_state_dict = _remap_gcp_encoder_keys(pretrained_state_dict, net, logger)

    if decoder_only:
        pretrained_state_dict = {
            k: v
            for k, v in pretrained_state_dict.items()
            if not (k.startswith('encoder') or k.startswith('vqvae.encoder'))
        }
    if drop_prefixes:
        prefixes = tuple(drop_prefixes) if isinstance(drop_prefixes, (list, tuple)) else (str(drop_prefixes),)
        pretrained_state_dict = {
            k: v for k, v in pretrained_state_dict.items()
            if not any(k.startswith(prefix) for prefix in prefixes)
        }

    load_log = net.load_state_dict(pretrained_state_dict, strict=False)
    logger.info(f'Loading checkpoint log: {load_log}')
    return net


def build_logger(result_dir, name="demo_evaluation"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_file_path = os.path.join(result_dir, "logs.txt")
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def record_indices(pids, indices_tensor, sequences, records, *, max_length=None):
    cpu_inds = indices_tensor.detach().cpu().tolist()
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    for pid, idx, seq in zip(pids, cpu_inds, sequences):
        if not isinstance(idx, list):
            idx = [idx]
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]
        records.append({'pid': pid, 'indices': [int(v) for v in idx[:len(seq)]], 'protein_sequence': seq})


def record_embeddings(pids, embeddings_array, indices_tensor, sequences, records, *, max_length=None):
    cpu_inds = indices_tensor.detach().cpu().tolist()
    for pid, emb, ind_list, seq in zip(pids, embeddings_array, cpu_inds, sequences):
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]
        emb_trim = emb[:len(seq)]
        ind_trim = ind_list[:len(seq)]
        records.append({
            'pid': pid,
            'embedding': emb_trim.astype('float32', copy=False),
            'indices': [int(v) for v in ind_trim],
            'protein_sequence': seq,
        })


def save_backbone_pdb_inference(
    coords,
    masks,
    save_path_prefix,
    atom_names=("N", "CA", "C"),
    chain_id="A",
):
    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
        masks = masks.unsqueeze(0)

    _, length = coords.shape[:2]

    for b in range(coords.shape[0]):
        out_path = save_path_prefix if save_path_prefix.lower().endswith('.pdb') else f"{save_path_prefix}.pdb"

        with open(out_path, "w") as fh:
            serial = 1
            for r in range(length):
                if masks[b, r].item() != 1:
                    continue

                for a_idx, atom_name in enumerate(atom_names):
                    if not torch.isfinite(coords[b, r, a_idx]).all():
                        continue

                    x, y, z = coords[b, r, a_idx].tolist()
                    element = atom_name[0].upper()

                    fh.write(
                        f"ATOM  "
                        f"{serial:5d} "
                        f"{atom_name:>4s}"
                        f" "
                        f"UNK"
                        f" "
                        f"{chain_id}"
                        f"{r + 1:4d}"
                        f" "
                        f"   "
                        f"{x:8.3f}"
                        f"{y:8.3f}"
                        f"{z:8.3f}"
                        f"{1.00:6.2f}"
                        f"{0.00:6.2f}"
                        f"          "
                        f"{element:>2s}"
                        "\n"
                    )
                    serial += 1

            fh.write("TER\nEND\n")


def save_predictions_to_pdb(pids, preds, masks, pdb_dir):
    for pid, coord, mask in zip(pids, preds, masks):
        prefix = os.path.join(pdb_dir, pid)
        save_backbone_pdb_inference(coord, mask, prefix)


def write_indices_csv(csv_path, indices_records):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pid', 'indices', 'protein_sequence'])
        for rec in indices_records:
            pid = rec['pid']
            inds = rec['indices']
            seq = rec['protein_sequence']
            if not isinstance(inds, (list, tuple)):
                inds = [inds]
            writer.writerow([pid, ' '.join(map(str, inds)), seq])


def write_embeddings_h5(h5_path, embeddings_records):
    with h5py.File(h5_path, 'w') as hf:
        for rec in embeddings_records:
            pid = rec['pid']
            emb = rec['embedding']
            inds = rec['indices']
            group = hf.create_group(pid)
            group.create_dataset('embedding', data=emb, compression='gzip')
            group.create_dataset('indices', data=np.array(inds, dtype=np.int32), compression='gzip')


def evaluate_structures(pdb_dir, original_pdb_dir, result_dir, logger, *, show_progress=True):
    logger.info("Starting TM-score and RMSD evaluation...")

    pred_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    if not pred_files:
        logger.warning("No PDB files found for evaluation")
        return

    results = []
    failed_evaluations = []

    for pred_file in tqdm(pred_files, desc="Evaluating structures", disable=not show_progress):
        pred_path = os.path.join(pdb_dir, pred_file)
        base_name = pred_file
        original_path = os.path.join(original_pdb_dir, base_name)

        if not os.path.exists(original_path):
            logger.warning(f"Original file not found for {pred_file}")
            failed_evaluations.append(pred_file)
            continue

        try:
            tm_scorer = TMscoring(pred_path, original_path)
            _, tm_score, rmsd = tm_scorer.optimise()
            tm_score = max(0.0, min(tm_score, 1.0))

            results.append({
                'pdb_file': os.path.splitext(pred_file)[0],
                'tm_score': tm_score,
                'rmsd': rmsd,
                'num_amino_acids': tm_scorer.N,
            })
        except Exception as exc:
            logger.error(f"Failed to evaluate {pred_file}: {str(exc)}")
            failed_evaluations.append(pred_file)

    if results:
        csv_path = os.path.join(result_dir, 'detailed_scores.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['pdb_file', 'tm_score', 'rmsd', 'num_amino_acids'])
            writer.writeheader()
            writer.writerows(results)

        tm_scores = [r['tm_score'] for r in results]
        rmsds = [r['rmsd'] for r in results]

        avg_tm_score = sum(tm_scores) / len(tm_scores)
        avg_rmsd = sum(rmsds) / len(rmsds)

        logger.info(f"Evaluation completed for {len(results)} structures")
        logger.info(f"Average TM-score: {avg_tm_score:.4f}")
        logger.info(f"Average RMSD: {avg_rmsd:.4f}")
        logger.info(f"Results saved to: {csv_path}")

        summary_path = os.path.join(result_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Structure Evaluation Summary\n")
            f.write("===========================\n\n")
            f.write(f"Total structures evaluated: {len(results)}\n")
            f.write(f"Failed evaluations: {len(failed_evaluations)}\n")
            f.write(f"Average TM-score: {avg_tm_score:.4f}\n")
            f.write(f"Average RMSD: {avg_rmsd:.4f}\n")
            f.write(f"TM-score range: {min(tm_scores):.4f} - {max(tm_scores):.4f}\n")
            f.write(f"RMSD range: {min(rmsds):.4f} - {max(rmsds):.4f}\n")

            if failed_evaluations:
                f.write("\nFailed evaluations:\n")
                for failed in failed_evaluations:
                    f.write(f"  - {failed}\n")
    else:
        logger.error("No structures were successfully evaluated")
