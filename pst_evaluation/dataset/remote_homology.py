import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

_FOLD45_CACHE = {"keep": None, "remap": None}


class RemoteHomologyDataset(Dataset):
    # Top 45 most frequent folds in training set (computed from current data)
    DEFAULT_FOLD_ALLOWLIST = [
        0,
        3,
        13,
        14,
        18,
        21,
        22,
        26,
        36,
        39,
        42,
        45,
        47,
        49,
        51,
        52,
        56,
        59,
        60,
        61,
        70,
        73,
        77,
        78,
        81,
        84,
        88,
        90,
        91,
        95,
        97,
        113,
        124,
        126,
        132,
        133,
        135,
        143,
        153,
        176,
        178,
        179,
        180,
        246,
        295,
    ]

    def __init__(
        self,
        data_path,
        split,
        pdb_data_dir,
        tokenizer,
        logger=None,
        cache=True,
        filter_length=600,
        fold_allowlist=None,
        fold_remap=None,
    ):
        self.data_path = data_path
        self.split = split
        self.tokenizer = tokenizer
        self.logger = logger
        self.cache = cache
        self.filter_length = int(filter_length) if filter_length is not None else None
        self._cache = {}

        # Map splits to file names
        split_map = {
            "train": "processed_structured_train",
            "validation": "processed_structured_valid",
            "fold_test": "processed_structured_test_fold_holdout",
            "superfamily_test": "processed_structured_test_superfamily_holdout",
            "family_test": "processed_structured_test_family_holdout",
        }
        file_name = split_map.get(split)
        if not file_name:
            raise ValueError(f"Unknown split: {split}")

        raw_file = os.path.join(data_path, "structural/remote_homology", file_name)
        if not os.path.exists(raw_file):
             # Fallback if structural/ is missing from data_path
             raw_file = os.path.join(data_path, "remote_homology", file_name)

        self._log(f"[data] loading {raw_file}")
        all_data = torch.load(raw_file, map_location="cpu", weights_only=False)

        keep_set, remap = self._resolve_fold_mapping(
            fold_allowlist=fold_allowlist,
            fold_remap=fold_remap,
            train_path=os.path.join(data_path, "structural/remote_homology", "processed_structured_train"),
        )
        self.num_classes = len(remap)

        # Filter to 45 classes + remap
        self.data = []
        for s in all_data:
            old_label = int(s.get("fold_label", -1))
            if old_label in keep_set:
                s["fold_label"] = remap[old_label]
                self.data.append(s)
        self._log(f"[data] loaded {len(self.data)} samples after 45-class filtering")
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

    def _resolve_fold_mapping(self, fold_allowlist, fold_remap, train_path):
        if fold_allowlist is not None:
            keep = list(map(int, fold_allowlist))
        else:
            if _FOLD45_CACHE["keep"] is None or _FOLD45_CACHE["remap"] is None:
                keep, remap = self._compute_top45(train_path)
                _FOLD45_CACHE["keep"] = keep
                _FOLD45_CACHE["remap"] = remap
            keep = _FOLD45_CACHE["keep"]
        if fold_remap is None:
            remap = _FOLD45_CACHE["remap"]
            if remap is None:
                keep_sorted = sorted(keep)
                remap = {old: new for new, old in enumerate(keep_sorted)}
        else:
            remap = fold_remap
        return set(keep), remap

    def _compute_top45(self, train_path):
        if not os.path.exists(train_path):
            keep_sorted = sorted(self.DEFAULT_FOLD_ALLOWLIST)
            remap = {old: new for new, old in enumerate(keep_sorted)}
            return keep_sorted, remap
        train_data = torch.load(train_path, map_location="cpu", weights_only=False)
        labels = [int(x.get("fold_label", -1)) for x in train_data if "fold_label" in x]
        counts = Counter(labels).most_common(45)
        keep_sorted = sorted([lab for lab, _ in counts])
        remap = {old: new for new, old in enumerate(keep_sorted)}
        return keep_sorted, remap

    def _prefilter_missing_h5(self):
        kept = []
        skipped = 0
        for item in self.data:
            pdb_id = str(item.get("pdb_id", "")).lower()
            chain_id = self._pick_chain_id(item.get("chain_id"))
            if not pdb_id:
                skipped += 1
                continue
            chain_up = str(chain_id).strip().upper() if chain_id is not None else ""
            arr, _ = self.tokenizer._read_from_h5(pdb_id, chain_up)
            if arr is None:
                skipped += 1
                continue
            kept.append(item)
        self.data = kept
        self._log(f"[data] kept {len(kept)} / {len(kept) + skipped} samples with H5 embeddings")

    def _pick_chain_id(self, chain_value):
        if isinstance(chain_value, (list, tuple, np.ndarray)):
            if len(chain_value) == 0:
                return None
            unique = []
            for c in chain_value:
                if c not in unique:
                    unique.append(c)
            if len(unique) != 1:
                return None
            return unique[0]
        return chain_value

    def _get_selected_indices_by_length(self, length, residue_range):
        """Select indices by position when residue_range is positional."""
        if residue_range == [""] or residue_range is None:
            return np.arange(length)

        # decide whether residue_range is 0-based or 1-based
        starts = []
        for r in residue_range:
            if "-" not in r:
                continue
            try:
                start, _ = map(int, r.split("-"))
                starts.append(start)
            except Exception:
                continue
        offset = 0 if (len(starts) > 0 and min(starts) == 0) else 1

        mask = np.zeros(length, dtype=bool)
        for r in residue_range:
            if "-" not in r:
                continue
            try:
                start, end = map(int, r.split("-"))
            except Exception:
                continue
            start_idx = max(0, start - offset)
            end_idx = min(length - 1, end - offset)
            if end_idx < start_idx:
                continue
            mask[start_idx : end_idx + 1] = True
        return np.where(mask)[0]

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        item = self.data[idx]
        pdb_id = str(item.get("pdb_id", "")).lower()
        chain_id = self._pick_chain_id(item.get("chain_id")) or "A"
        residue_range = item.get("residue_range", [""])

        arr, _ = self.tokenizer._read_from_h5(pdb_id, str(chain_id).strip().upper())
        if arr is None:
            return None
        if arr.ndim == 1:
            arr = arr[None, :]
        output_dtype = getattr(self.tokenizer, "output_dtype", torch.float32)
        token_ids = torch.as_tensor(arr, dtype=output_dtype)
        # Crop to domain by positional residue ranges
        selected_indices = self._get_selected_indices_by_length(int(token_ids.shape[0]), residue_range)
        
        if len(selected_indices) == 0:
            return None
            
        token_ids = token_ids[selected_indices]
        if self.filter_length is not None:
            try:
                if int(token_ids.shape[0]) > self.filter_length:
                    return None
            except Exception:
                return None

        # Remote Homology is global classification
        out = {
            "token_ids": token_ids.detach().cpu(),
            "label": torch.tensor(item["fold_label"], dtype=torch.long),
        }
        
        if self.cache:
            self._cache[idx] = out
        return out

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0: return None
        
        feats_list = [b["token_ids"] for b in batch]
        labels_list = [b["label"] for b in batch]
        lengths = [int(x.shape[0]) for x in feats_list]
        Lmax = max(lengths)
        D = max(int(x.shape[1]) for x in feats_list)
        B = len(batch)

        feats = torch.zeros((B, Lmax, D), dtype=torch.float32)
        attn_mask = torch.ones((B, Lmax), dtype=torch.bool)
        
        for i, x in enumerate(feats_list):
            L = x.shape[0]
            feats[i, :L, :] = x
            attn_mask[i, :L] = False
            
        return {
            "token_ids": feats,
            "labels": torch.stack(labels_list),
            "attention_mask": attn_mask,
            "lengths": torch.as_tensor(lengths, dtype=torch.int32),
        }
