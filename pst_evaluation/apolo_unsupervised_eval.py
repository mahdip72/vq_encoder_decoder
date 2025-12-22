#!/usr/bin/env python3
import os
import argparse
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef
from biotite.sequence import Alphabet, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from dataset.conformational_switch import ConformationalSwitchDataset
from dataset.tokenizer_biolip2 import WrappedMyRepBioLIP2Tokenizer


def _iter_with_progress(loader, enabled, desc):
    if not enabled:
        return loader
    try:
        from tqdm import tqdm

        return tqdm(loader, desc=desc, leave=False)
    except Exception:
        return loader


def compute_alignment_score(
    feats1: torch.Tensor,
    feats2: torch.Tensor,
    gap_penalty: int,
    terminal_penalty: bool,
    local_align: bool,
    sim_scale: float,
    sim_clip: float | None,
    score_norm: str,
    l2_normalize: bool,
) -> float:
    if l2_normalize:
        feats1 = F.normalize(feats1, p=2, dim=-1)
        feats2 = F.normalize(feats2, p=2, dim=-1)
    sim = torch.matmul(feats1, feats2.T)  # [L1, L2]
    if sim_clip is not None:
        sim = torch.clamp(sim, -sim_clip, sim_clip)
    sim = (sim.detach().cpu().numpy() * sim_scale).astype(np.int32)
    L1, L2 = sim.shape
    sim_score = np.zeros((L1 + L2, L1 + L2), dtype=np.int32)
    sim_score[:L1, L1:] = sim
    alphabet = Alphabet(list(range(L1 + L2)))
    substitution_matrix = SubstitutionMatrix(alphabet, alphabet, sim_score)
    seq1 = GeneralSequence(alphabet, np.arange(L1))
    seq2 = GeneralSequence(alphabet, np.arange(L2) + L1)
    align_score = align_optimal(
        seq1,
        seq2,
        substitution_matrix,
        gap_penalty=gap_penalty,
        terminal_penalty=terminal_penalty,
        local=local_align,
    )[0].score
    score = float(align_score)
    if score_norm == "min":
        denom = float(min(L1, L2))
    elif score_norm == "max":
        denom = float(max(L1, L2))
    elif score_norm == "mean":
        denom = float((L1 + L2) / 2.0)
    elif score_norm == "sum":
        denom = float(L1 + L2)
    elif score_norm == "geo":
        denom = float(math.sqrt(L1 * L2))
    else:
        denom = 1.0
    if denom > 0:
        score = score / denom
    return score


@torch.no_grad()
def eval_unsupervised(model_device, loader, align_cfg, show_progress=False, desc="eval"):
    preds = []
    targets = []
    for batch in _iter_with_progress(loader, show_progress, desc=desc):
        if batch is None:
            continue
        prot1_list = batch["prot1"]
        prot2_list = batch["prot2"]
        labels = batch["labels"]
        for feats1, feats2, label in zip(prot1_list, prot2_list, labels):
            feats1 = feats1.to(model_device)
            feats2 = feats2.to(model_device)
            if feats1.numel() == 0 or feats2.numel() == 0:
                continue
            score = compute_alignment_score(
                feats1,
                feats2,
                gap_penalty=align_cfg["gap_penalty"],
                terminal_penalty=align_cfg["terminal_penalty"],
                local_align=align_cfg["local_align"],
                sim_scale=align_cfg["sim_scale"],
                sim_clip=align_cfg["sim_clip"],
                score_norm=align_cfg["score_norm"],
                l2_normalize=align_cfg["l2_normalize"],
            )
            preds.append(score)
            targets.append(float(label))

    if len(preds) < 2:
        return float("nan"), float("nan"), len(preds)

    preds_t = torch.tensor(preds, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)
    spearman = SpearmanCorrCoef()
    pearson = PearsonCorrCoef()
    spearman_score = float(spearman(preds_t, targets_t).detach().cpu().item())
    pearson_score = float(pearson(preds_t, targets_t).detach().cpu().item())
    return spearman_score, pearson_score, len(preds)


def parse_args():
    p = argparse.ArgumentParser(description="Apo/Holo and Fold Switching unsupervised evaluation (minimal)")
    p.add_argument(
        "--h5",
        required=True,
        help="Path to apo/holo embedding H5 (e.g., vq_embed_apolo_lite.h5)",
    )
    p.add_argument(
        "--embeddings-dataset",
        default="/",
        help="H5 dataset/group path (default: '/')",
    )
    p.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(__file__), "struct_token_bench_release_data", "data", "sensitivity"),
        help="Root containing conformational/processed_structured_* files",
    )
    p.add_argument(
        "--target-field",
        default="tm_score",
        choices=["tm_score", "negrmsd_score"],
        help="Target field to compare against",
    )
    p.add_argument("--gap-penalty", type=int, default=-10)
    p.add_argument("--local", action="store_true", help="Use local alignment instead of global")
    p.add_argument("--no-terminal-penalty", action="store_true", help="Disable terminal gap penalties")
    p.add_argument("--sim-scale", type=float, default=100.0)
    p.add_argument("--sim-clip", type=float, default=0.0, help="Clip cosine similarity before scaling (0 disables)")
    p.add_argument(
        "--score-norm",
        default="none",
        choices=["none", "min", "max", "mean", "sum", "geo"],
        help="Normalize alignment score by sequence lengths",
    )
    p.add_argument("--no-l2-normalize", action="store_true", help="Disable L2 normalization")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--filter-length", type=int, default=600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer-device", default="cpu")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("apolo_unsupervised_eval")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    tokenizer = WrappedMyRepBioLIP2Tokenizer(
        h5_path=args.h5,
        embeddings_dataset=args.embeddings_dataset,
        device=str(args.tokenizer_device),
    )

    align_cfg = {
        "gap_penalty": args.gap_penalty,
        "terminal_penalty": not args.no_terminal_penalty,
        "local_align": args.local,
        "sim_scale": float(args.sim_scale),
        "sim_clip": float(args.sim_clip) if args.sim_clip and args.sim_clip > 0 else None,
        "score_norm": args.score_norm,
        "l2_normalize": not args.no_l2_normalize,
    }

    splits = ["apo_holo_test", "fold_switching_test"]
    results = {}
    for split in splits:
        logger.info(f"[data] target_field={args.target_field} split={split}")
        ds = ConformationalSwitchDataset(
            data_path=args.data_root,
            split=split,
            target_field=args.target_field,
            tokenizer=tokenizer,
            logger=logger,
            cache=True,
            filter_length=args.filter_length,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=ds.collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
        )

        logger.info("[eval] start")
        spearman, pearson, n = eval_unsupervised(
            device, loader, align_cfg, show_progress=args.progress, desc=split
        )
        results[split] = (spearman, pearson, n)
        logger.info(f"[eval] split={split} n={n} spearman={spearman:.4f} pearson={pearson:.4f}")

    apo_spear, apo_pear, apo_n = results.get("apo_holo_test", (float("nan"), float("nan"), 0))
    fold_spear, fold_pear, fold_n = results.get("fold_switching_test", (float("nan"), float("nan"), 0))
    logger.info(
        "[table] "
        f"apo_holo PCC%={apo_pear*100:.2f} Spearman%={apo_spear*100:.2f} (n={apo_n}) | "
        f"fold_switching PCC%={fold_pear*100:.2f} Spearman%={fold_spear*100:.2f} (n={fold_n})"
    )


if __name__ == "__main__":
    main()
