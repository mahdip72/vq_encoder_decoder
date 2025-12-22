#!/usr/bin/env python3
import os
import argparse
import copy
import logging
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC

from dataset.biolip2 import BioLIP2FunctionDataset
from dataset.tokenizer_biolip2 import WrappedMyRepBioLIP2Tokenizer


class MLPHead(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layer=1, dropout=0.0):
        super().__init__()
        dims = [in_dim] + [hid_dim] * num_layer + [out_dim]
        layers = []
        for i in range(num_layer + 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layer:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.classify = nn.Sequential(*layers)

    def forward(self, x):
        return self.classify(x)


class BindingPredictor(nn.Module):
    def __init__(self, embed_dim, d_model, hidden_size, num_layer, dropout):
        super().__init__()
        if d_model is None:
            d_model = embed_dim
        self.d_model = int(d_model)
        self.proj = None
        if embed_dim != self.d_model:
            self.proj = nn.Linear(embed_dim, self.d_model, bias=False)
        self.head = MLPHead(self.d_model, hidden_size, 1, num_layer=num_layer, dropout=dropout)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        return self.head(x)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=0.5,
    last_epoch=-1,
    min_ratio=0.1,
    plateau_ratio=0.1,
):
    def lr_lambda(current_step):
        plateau_steps = int(plateau_ratio * num_training_steps)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + plateau_steps:
            return 1.0
        progress = float(current_step - num_warmup_steps - plateau_steps) / float(
            max(1, num_training_steps - num_warmup_steps - plateau_steps)
        )
        return max(
            min_ratio,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def build_dataset(split: str, args, tokenizer, logger):
    logger.info(f"[data] init dataset split={split}")
    ds = BioLIP2FunctionDataset(
        data_path=args.data_root,
        split=split,
        target_field="binding_label",
        pdb_data_dir=None,
        tokenizer=tokenizer,
        logger=logger,
        cache=True,
        filter_length=args.filter_length,
    )
    logger.info(f"[data] dataset split={split} size={len(ds)}")
    return ds


def make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def _iter_with_progress(loader, enabled, desc):
    if not enabled:
        return loader
    try:
        from tqdm import tqdm
        return tqdm(loader, desc=desc, leave=False)
    except Exception:
        return loader


def train_one_epoch(model, loader, optimizer, device, show_progress=False):
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in _iter_with_progress(loader, show_progress, desc="train"):
        if batch is None:
            continue
        feats = batch["token_ids"].to(device)
        targets = batch["labels"].to(device)
        mask = targets != -100
        if mask.sum() == 0:
            continue
        logits = model(feats).squeeze(-1)
        logits = logits[mask]
        targets = targets[mask]
        pos = targets.sum()
        neg = targets.numel() - pos
        if pos > 0 and neg > 0:
            pos_weight = targets.numel() / pos * 0.5
            neg_weight = targets.numel() / neg * 0.5
            weights = torch.full_like(targets, neg_weight)
            weights[targets >= 0.5] = pos_weight
            loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(1, steps)


@torch.no_grad()
def eval_auroc(model, loader, device, show_progress=False, desc="eval"):
    model.eval()
    metric = AUROC(task="binary").to(device)
    total = 0
    pos = 0
    for batch in _iter_with_progress(loader, show_progress, desc=desc):
        if batch is None:
            continue
        feats = batch["token_ids"].to(device)
        targets = batch["labels"].to(device)
        mask = targets != -100
        if mask.sum() == 0:
            continue
        logits = model(feats).squeeze(-1)
        scores = torch.sigmoid(logits[mask])
        labs = targets[mask]
        total += int(labs.numel())
        pos += int(labs.sum().item())
        metric.update(scores, labs.to(torch.int64))
    if total == 0 or pos == 0 or pos == total:
        return float("nan")
    return float(metric.compute().detach().cpu().item())


def parse_args():
    p = argparse.ArgumentParser(description="BioLIP2 binding site AUROC eval (minimal)")
    p.add_argument(
        "--h5",
        required=True,
        help="Path to BioLIP2 embedding H5 (e.g., vq_embed_biolip2_binding_lite_model.h5)",
    )
    p.add_argument(
        "--embeddings-dataset",
        default="/",
        help="H5 dataset/group path (default: '/')",
    )
    p.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(__file__), "struct_token_bench_release_data", "data", "functional", "local"),
        help="Root containing biolip2/processed_structured_* files",
    )
    p.add_argument("--epochs", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--filter-length", type=int, default=600)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layer", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--plateau-ratio", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer-device", default="cpu")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bars")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("biolip2_binding_eval")

    seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    tokenizer = WrappedMyRepBioLIP2Tokenizer(
        h5_path=args.h5,
        embeddings_dataset=args.embeddings_dataset,
        device=str(args.tokenizer_device),
    )

    train_ds = build_dataset("train", args, tokenizer, logger)
    val_ds = build_dataset("validation", args, tokenizer, logger)
    fold_ds = build_dataset("fold_test", args, tokenizer, logger)
    sfam_ds = build_dataset("superfamily_test", args, tokenizer, logger)

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    fold_loader = make_loader(fold_ds, args.batch_size, args.num_workers, shuffle=False)
    sfam_loader = make_loader(sfam_ds, args.batch_size, args.num_workers, shuffle=False)

    sample_dim = None
    for batch in train_loader:
        if batch is None:
            continue
        sample_dim = int(batch["token_ids"].shape[-1])
        break
    if sample_dim is None:
        raise RuntimeError("No valid samples in training set.")
    embed_dim = sample_dim
    model = BindingPredictor(
        embed_dim=embed_dim,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_layer=args.num_layer,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        min_ratio=args.min_lr_ratio,
        plateau_ratio=args.plateau_ratio,
    )

    logger.info("[train] start")
    best_state = None
    best_val = float("-inf")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in _iter_with_progress(train_loader, args.progress, desc="train"):
            if batch is None:
                continue
            feats = batch["token_ids"].to(device)
            targets = batch["labels"].to(device)
            mask = targets != -100
            if mask.sum() == 0:
                continue
            logits = model(feats).squeeze(-1)
            logits = logits[mask]
            targets = targets[mask]
            pos = targets.sum()
            neg = targets.numel() - pos
            if pos > 0 and neg > 0:
                pos_weight = targets.numel() / pos * 0.5
                neg_weight = targets.numel() / neg * 0.5
                weights = torch.full_like(targets, neg_weight)
                weights[targets >= 0.5] = pos_weight
                loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            steps += 1
            global_step += 1
        loss = total_loss / max(1, steps)
        val_auroc = eval_auroc(model, val_loader, device, show_progress=args.progress, desc="val")
        fold_auroc = eval_auroc(model, fold_loader, device, show_progress=args.progress, desc="fold_test")
        sfam_auroc = eval_auroc(model, sfam_loader, device, show_progress=args.progress, desc="superfamily_test")
        logger.info(
            f"[train] epoch={epoch+1} loss={loss:.4f} "
            f"val_auroc={val_auroc*100:.2f} "
            f"fold_test_auroc={fold_auroc*100:.2f} "
            f"superfamily_test_auroc={sfam_auroc*100:.2f}"
        )
        if val_auroc > best_val:
            best_val = val_auroc
            best_state = copy.deepcopy(model.state_dict())
    logger.info("[train] finished")

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"[train] loaded best val checkpoint (val_auroc={best_val*100:.2f})")

    fold_auroc = eval_auroc(model, fold_loader, device, show_progress=args.progress, desc="fold_test")
    sfam_auroc = eval_auroc(model, sfam_loader, device, show_progress=args.progress, desc="superfamily_test")
    logger.info(f"[final] fold_test AUROC% = {fold_auroc*100:.2f}")
    logger.info(f"[final] superfamily_test AUROC% = {sfam_auroc*100:.2f}")


if __name__ == "__main__":
    main()
