import os

import torch
from torch.utils.data import Dataset

from .tokenizer_biolip2 import MissingRepresentation


class ConformationalSwitchDataset(Dataset):
    def __init__(
        self,
        data_path,
        split,
        target_field,
        tokenizer,
        logger=None,
        cache=True,
        filter_length=600,
    ):
        self.data_path = data_path
        self.split = split
        self.target_field = target_field
        self.tokenizer = tokenizer
        self.logger = logger
        self.cache = bool(cache)
        self.filter_length = int(filter_length) if filter_length is not None else None
        self._cache = {}

        raw_file = os.path.join(
            self.data_path,
            "conformational",
            f"processed_structured_{self.target_field}_{self.split}",
        )
        if not os.path.isfile(raw_file):
            raise FileNotFoundError(f"Processed conformational file not found: {raw_file}")
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
            ok = True
            for idx in (1, 2):
                pdb_id = str(item.get(f"prot{idx}_pdb_id", "")).lower()
                chain_id = item.get(f"prot{idx}_chain_id")
                chain_up = str(chain_id).strip().upper() if chain_id is not None else ""
                if not pdb_id:
                    ok = False
                    break
                arr, _ = self.tokenizer._read_from_h5(pdb_id, chain_up)
                if arr is None:
                    ok = False
                    break
            if ok:
                kept.append(item)
            else:
                skipped += 1
        self.data = kept
        self._log(f"[data] kept {len(kept)} / {len(kept) + skipped} samples with H5 embeddings")

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        item = self.data[idx]
        outputs = []
        for i in (1, 2):
            pdb_id = str(item.get(f"prot{i}_pdb_id", "")).lower()
            chain_id = item.get(f"prot{i}_chain_id")
            if not pdb_id:
                return None
            chain_str = str(chain_id) if chain_id is not None else ""
            try:
                token_ids, _, _ = self.tokenizer.encode_structure(
                    f"{pdb_id}.cif", chain_str, use_sequence=False
                )
            except MissingRepresentation:
                return None
            except Exception:
                return None
            if token_ids.numel() == 0:
                return None
            if self.filter_length is not None:
                try:
                    if int(token_ids.shape[0]) > self.filter_length:
                        return None
                except Exception:
                    return None
            outputs.append(token_ids.detach().cpu())

        label = item.get(self.target_field)
        if label is None:
            return None

        out = {
            "prot1": outputs[0],
            "prot2": outputs[1],
            "label": float(label),
        }
        if self.cache:
            self._cache[idx] = out
        return out

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        prot1_list = [b["prot1"] for b in batch]
        prot2_list = [b["prot2"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        return {
            "prot1": prot1_list,
            "prot2": prot2_list,
            "labels": labels,
        }
