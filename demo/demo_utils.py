import csv
import logging
import os
import sys

import h5py
import numpy as np
import yaml
from box import Box
from tqdm import tqdm

from utils.utils import save_backbone_pdb_inference
from utils.evaluation.tmscore import TMscoring


def load_saved_encoder_decoder_configs(encoder_cfg_path, decoder_cfg_path):
    with open(encoder_cfg_path) as f:
        enc_cfg = yaml.full_load(f)
    encoder_configs = Box(enc_cfg)

    with open(decoder_cfg_path) as f:
        dec_cfg = yaml.full_load(f)
    decoder_configs = Box(dec_cfg)

    return encoder_configs, decoder_configs


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


def record_indices(pids, indices_tensor, sequences, records):
    cpu_inds = indices_tensor.detach().cpu().tolist()
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    for pid, idx, seq in zip(pids, cpu_inds, sequences):
        if not isinstance(idx, list):
            idx = [idx]
        cleaned = [int(v) for v in idx if v != -1]
        records.append({'pid': pid, 'indices': cleaned, 'protein_sequence': seq})


def record_embeddings(pids, embeddings_array, indices_tensor, sequences, records):
    cpu_inds = indices_tensor.detach().cpu().tolist()
    for pid, emb, ind_list, seq in zip(pids, embeddings_array, cpu_inds, sequences):
        seq_len = len(seq)
        emb_trim = emb[:seq_len]
        ind_trim = ind_list[:seq_len]
        cleaned = [int(v) for v in ind_trim if v != -1]
        if len(cleaned) != len(ind_trim):
            keep_positions = [i for i, v in enumerate(ind_trim) if v != -1]
            emb_trim = emb_trim[keep_positions]
        ind_trim = cleaned
        records.append({
            'pid': pid,
            'embedding': emb_trim.astype('float32', copy=False),
            'indices': ind_trim,
            'protein_sequence': seq,
        })


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
