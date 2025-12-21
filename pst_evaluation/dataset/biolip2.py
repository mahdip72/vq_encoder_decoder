import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer_biolip2 import MissingRepresentation


def parse_binding_indices(residue_str):
    if residue_str is None:
        return None
    if isinstance(residue_str, float) and math.isnan(residue_str):
        return None
    if isinstance(residue_str, str) and residue_str.strip() == "?":
        return None

    idx_set = set()
    parts = str(residue_str).split(" ")
    for tok in parts:
        tok = tok.strip()
        if not tok:
            continue
        ri = tok[1:]
        if not ri:
            continue
        if not ri[-1].isdigit():
            ri = ri[:-1]
        if not ri:
            continue
        try:
            idx_set.add(int(ri))
        except Exception:
            continue
    return idx_set


class BioLIP2FunctionDataset(Dataset):
    FULL_FIELD_MAPPING = {
        "binding_label": "binding_site_residues_pdb_numbered",
        "catalytic_label": "catalytic_site_residues_pdb_numbered",
    }

    def __init__(
        self,
        data_path,
        split,
        target_field,
        pdb_data_dir,
        tokenizer,
        logger=None,
        cache=True,
        filter_length=600,
    ):
        self.data_path = data_path
        self.split = split
        self.target_field = target_field
        self.pdb_data_dir = pdb_data_dir
        self.tokenizer = tokenizer
        self.logger = logger
        self.cache = bool(cache)
        self.filter_length = int(filter_length) if filter_length is not None else None
        self._cache = {}

        raw_file = os.path.join(
            self.data_path,
            "biolip2",
            f"{self.target_field}_{self.split}",
        )
        self.data = torch.load(raw_file, map_location="cpu", weights_only=False)
        self._prefilter_missing_h5()

    def __len__(self):
        return len(self.data)

    def _log(self, msg):
        if self.logger is not None:
            try:
                self.logger.info(msg)
                return
            except Exception:
                pass
        print(msg)

    def _prefilter_missing_h5(self):
        kept = []
        skipped = 0
        for item in self.data:
            pdb_id = str(item.get("pdb_id", "")).lower()
            chain_id = item.get("receptor_chain") or item.get("chain_id")
            if not pdb_id or chain_id is None:
                skipped += 1
                continue
            arr, _ = self.tokenizer._read_from_h5(pdb_id, str(chain_id).strip().upper())
            if arr is None:
                skipped += 1
                continue
            kept.append(item)
        self.data = kept
        self._log(f"[data] kept {len(kept)} / {len(kept) + skipped} samples with H5 embeddings")

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        item = self.data[idx]
        pdb_id = str(item.get("pdb_id", "")).lower()
        chain_id = item.get("receptor_chain") or item.get("chain_id")
        if not pdb_id or chain_id is None:
            return None

        label_field = self.FULL_FIELD_MAPPING.get(self.target_field)
        if label_field is None:
            return None
        residue_str = item.get(label_field, None)
        idx_set = parse_binding_indices(residue_str)
        if idx_set is None:
            return None

        try:
            token_ids, token_res_idx, _ = self.tokenizer.encode_structure(
                f"{pdb_id}.cif", str(chain_id), use_sequence=False
            )
        except MissingRepresentation:
            return None
        except Exception:
            return None
        if self.filter_length is not None:
            try:
                if int(token_ids.shape[0]) > self.filter_length:
                    return None
            except Exception:
                return None

        token_res_idx = np.asarray(token_res_idx)
        if token_res_idx.size == 0:
            return None
        labels = np.asarray([1 if int(r) in idx_set else 0 for r in token_res_idx], dtype=np.float32)
        if len(token_ids) != len(labels):
            return None

        out = {
            "token_ids": token_ids.detach().cpu(),
            "label": torch.as_tensor(labels, dtype=torch.float32),
        }
        if self.cache:
            self._cache[idx] = out
        return out

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None

        feats_list = [b["token_ids"] for b in batch]
        labels_list = [b["label"] for b in batch]
        lengths = [int(x.shape[0]) for x in feats_list]
        Lmax = max(lengths)
        D = max(int(x.shape[1]) for x in feats_list)
        B = len(batch)

        feats = torch.zeros((B, Lmax, D), dtype=torch.float32)
        targets = torch.full((B, Lmax), fill_value=-100, dtype=torch.float32)
        attn = torch.ones((B, Lmax), dtype=torch.bool)

        for i, (x, y) in enumerate(zip(feats_list, labels_list)):
            L = int(x.shape[0])
            feats[i, :L, : x.shape[1]] = x
            targets[i, :L] = y[:L]
            attn[i, :L] = False

        return {
            "token_ids": feats,
            "labels": targets,
            "attention_mask": attn,
            "lengths": torch.as_tensor(lengths, dtype=torch.int32),
        }
