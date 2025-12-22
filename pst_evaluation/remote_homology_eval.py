#!/usr/bin/env python3
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
from torchmetrics.classification import MulticlassF1Score

from dataset.remote_homology import RemoteHomologyDataset
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


class ClassificationPredictor(nn.Module):
    def __init__(self, embed_dim, d_model, hidden_size, num_layer, dropout, num_classes):
        super().__init__()
        if d_model is None:
            d_model = embed_dim
        self.d_model = int(d_model)
        self.proj = None
        if embed_dim != self.d_model:
            self.proj = nn.Linear(embed_dim, self.d_model, bias=False)
        self.head = MLPHead(self.d_model, hidden_size, num_classes, num_layer=num_layer, dropout=dropout)

    def forward(self, x, attention_mask=None):
        if self.proj is not None:
            x = self.proj(x)
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = (~attention_mask).unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (x * mask).sum(dim=1) / denom
        return self.head(pooled)


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
    min_ratio=0.01,
    plateau_ratio=0.0,
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
    ds = RemoteHomologyDataset(
        data_path=args.data_root,
        split=split,
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


@torch.no_grad()
def eval_macro_f1(model, loader, device, num_classes, show_progress=False, desc="eval"):
    model.eval()
    metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    total = 0
    for batch in _iter_with_progress(loader, show_progress, desc=desc):
        if batch is None:
            continue
        feats = batch["token_ids"].to(device)
        targets = batch["labels"].to(device)
        attn_mask = batch.get("attention_mask")
        logits = model(feats, attention_mask=attn_mask.to(device) if attn_mask is not None else None)
        preds = torch.argmax(logits, dim=-1)
        metric.update(preds, targets)
        total += int(targets.numel())
    if total == 0:
        return float("nan")
    return float(metric.compute().detach().cpu().item())


def parse_args():
    p = argparse.ArgumentParser(description="Remote homology macro-F1 eval (minimal)")
    p.add_argument(
        "--h5",
        required=True,
        help="Path to structural embedding H5",
    )
    p.add_argument(
        "--embeddings-dataset",
        default="/",
        help="H5 dataset/group path",
    )
    p.add_argument(
        "--data-root",
        default="/mnt/hdd8/farzaneh/projects/PST/struct_token_bench_release_data/data/structural",
        help="Root containing structural/remote_homology/processed_structured_* files",
    )
    p.add_argument("--epochs", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=8)
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
    logger = logging.getLogger("remote_homology_eval")

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
    fam_ds = build_dataset("family_test", args, tokenizer, logger)

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    fold_loader = make_loader(fold_ds, args.batch_size, args.num_workers, shuffle=False)
    sfam_loader = make_loader(sfam_ds, args.batch_size, args.num_workers, shuffle=False)
    fam_loader = make_loader(fam_ds, args.batch_size, args.num_workers, shuffle=False)

    sample_dim = None
    for batch in train_loader:
        if batch is None:
            continue
        sample_dim = int(batch["token_ids"].shape[-1])
        break
    if sample_dim is None:
        raise RuntimeError("No valid samples in training set.")

    num_classes = getattr(train_ds, "num_classes", 45)
    model = ClassificationPredictor(
        embed_dim=sample_dim,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_layer=args.num_layer,
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
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
    best_val_f1 = float("-inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch in _iter_with_progress(train_loader, args.progress, desc="train"):
            if batch is None:
                continue
            feats = batch["token_ids"].to(device)
            targets = batch["labels"].to(device)
            attn_mask = batch.get("attention_mask")

            logits = model(feats, attention_mask=attn_mask.to(device) if attn_mask is not None else None)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        train_f1 = eval_macro_f1(
            model, train_loader, device, num_classes=num_classes, show_progress=args.progress, desc="train_eval"
        )
        val_f1 = eval_macro_f1(
            model, val_loader, device, num_classes=num_classes, show_progress=args.progress, desc="val"
        )
        logger.info(
            f"[train] epoch={epoch+1} loss={avg_loss:.4f} train_f1={train_f1*100:.2f} val_f1={val_f1*100:.2f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    logger.info("[train] finished")

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"[train] loaded best val checkpoint (val_f1={best_val_f1*100:.2f})")

    fold_f1 = eval_macro_f1(
        model, fold_loader, device, num_classes=num_classes, show_progress=args.progress, desc="fold_test"
    )
    sfam_f1 = eval_macro_f1(
        model, sfam_loader, device, num_classes=num_classes, show_progress=args.progress, desc="sfam_test"
    )
    fam_f1 = eval_macro_f1(
        model, fam_loader, device, num_classes=num_classes, show_progress=args.progress, desc="fam_test"
    )

    logger.info(f"[final] fold_test f1={fold_f1*100:.2f}")
    logger.info(f"[final] superfamily_test f1={sfam_f1*100:.2f}")
    logger.info(f"[final] family_test f1={fam_f1*100:.2f}")


if __name__ == "__main__":
    main()
