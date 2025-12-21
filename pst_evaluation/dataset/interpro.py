import os

import numpy as np
import torch
from torch.utils.data import Dataset
from biotite.sequence import Alphabet, GeneralSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from .tokenizer_biolip2 import MissingRepresentation


class InterProFunctionDataset(Dataset):
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
            "interpro",
            f"processed_structured_{self.target_field}_{self.split}",
        )
        if not os.path.isfile(raw_file):
            raise FileNotFoundError(f"Processed InterPro file not found: {raw_file}")
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
            chain_id = item.get("chain_id")
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

    def _align_by_residue_index(self, token_ids, token_res_idx, label_res_idx, labels):
        token_res_idx = np.asarray(token_res_idx)
        label_res_idx = np.asarray(label_res_idx)
        labels = np.asarray(labels, dtype=np.float32)

        align_idx1 = [i for i, x in enumerate(label_res_idx) if x in token_res_idx]
        if len(align_idx1) == 0:
            return None
        label_res_idx = label_res_idx[align_idx1]
        labels = labels[align_idx1]

        align_idx2 = [i for i, x in enumerate(token_res_idx) if x in label_res_idx]
        if len(align_idx2) == 0:
            return None
        token_res_idx = token_res_idx[align_idx2]
        token_ids = token_ids[align_idx2]

        need_align = (
            len(token_res_idx) != len(label_res_idx)
            or not np.array_equal(token_res_idx, np.array(label_res_idx))
        )
        if need_align:
            idx_list = list(set(token_res_idx.tolist() + label_res_idx.tolist()))
            alphabet = Alphabet(idx_list)
            sim_score = np.eye(len(idx_list), dtype=np.int32)
            substitution_matrix = SubstitutionMatrix(alphabet, alphabet, sim_score)
            seq1 = GeneralSequence(alphabet, label_res_idx)
            seq2 = GeneralSequence(alphabet, token_res_idx.tolist())
            alignment = align_optimal(seq1, seq2, substitution_matrix)[0].trace
            align_idx1, align_idx2 = [], []
            for i in range(len(alignment)):
                if (alignment[i] != -1).all():
                    align_idx1.append(alignment[i][0])
                    align_idx2.append(alignment[i][1])
            label_res_idx = label_res_idx[align_idx1]
            labels = labels[align_idx1]
            token_res_idx = token_res_idx[align_idx2]
            token_ids = token_ids[align_idx2]

            if not (len(token_res_idx) == len(label_res_idx) == len(labels)):
                return None

        if len(token_ids) == 0:
            return None
        return token_ids, labels

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            return self._cache[idx]

        item = self.data[idx]
        pdb_id = str(item.get("pdb_id", "")).lower()
        chain_id = item.get("chain_id")
        if not pdb_id or chain_id is None:
            return None

        labels = item.get(self.target_field)
        pdb_chain = item.get("pdb_chain")
        if labels is None or pdb_chain is None:
            return None
        label_res_idx = getattr(pdb_chain, "residue_index", None)
        if label_res_idx is None:
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

        if len(token_ids) == len(labels):
            labels = np.asarray(labels, dtype=np.float32)
        else:
            aligned = self._align_by_residue_index(token_ids, token_res_idx, label_res_idx, labels)
            if aligned is None:
                return None
            token_ids, labels = aligned

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
