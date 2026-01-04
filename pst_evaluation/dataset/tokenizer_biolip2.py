import os

import h5py
import numpy as np
import torch


class MissingRepresentation(Exception):
    pass


class WrappedMyRepBioLIP2Tokenizer:
    pad_token_id = 0

    def __init__(
        self,
        h5_path=None,
        embeddings_dataset="/",
        embed_dim=128,
        fallback_to_any_chain=False,
        skip_on_missing=True,
        device="cpu",
        output_dtype=None,
    ):
        self.device = device
        self.embed_dim = int(embed_dim)
        self.fallback = bool(fallback_to_any_chain)
        self.skip_on_missing = bool(skip_on_missing)
        if output_dtype is None:
            output_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        elif isinstance(output_dtype, str):
            output_dtype = getattr(torch, output_dtype)
        self.output_dtype = output_dtype
        self._last_debug = {}

        if not h5_path:
            raise ValueError("h5_path is required")
        self.h5_path = os.path.abspath(os.path.expanduser(h5_path))
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"H5 not found: {self.h5_path}")

        self.h5 = h5py.File(self.h5_path, "r")
        ds = embeddings_dataset if isinstance(embeddings_dataset, str) else "/"
        if ds == "/" or ds == "":
            self.emb = self.h5
        elif ds in self.h5:
            self.emb = self.h5[ds]
        else:
            obj = self.h5.get(ds)
            if isinstance(obj, (h5py.Group, h5py.Dataset)):
                self.emb = obj
            else:
                self.emb = self.h5

        self._infer_embed_dim()
        print(
            f"[tokenizer] loaded h5={self.h5_path} dataset={ds} embed_dim={self.embed_dim}"
        )

    def _infer_embed_dim(self):
        try:
            sample = None
            if isinstance(self.emb, h5py.Group):
                first_key = next(iter(self.emb.keys()))
                sample = self._read_group(self.emb[first_key])[0]
            elif isinstance(self.emb, h5py.Dataset):
                sample = self.emb[()]
            if sample is not None:
                if sample.ndim == 1:
                    sample = sample[None, :]
                self.embed_dim = int(sample.shape[-1])
        except Exception:
            pass

    def get_num_tokens(self):
        return None

    def _candidate_keys(self, pdb_id, chain_up):
        base = pdb_id.lower()
        bases = [base, base.upper()]
        cands = []
        for b in bases:
            if chain_up:
                chain_low = chain_up.lower()
                cands += [
                    f"{b}_chain_id_{chain_up}",
                    f"{b}_CHAIN_ID_{chain_up}",
                    f"{b}_chain_id_{chain_low}",
                    f"{b}_CHAIN_ID_{chain_low}",
                    f"{b}_{chain_up}",
                    f"{b}_{chain_low}",
                    f"{b}{chain_up}",
                    f"{b}{chain_low}",
                ]
            cands.append(b)
        return cands

    def _read_group(self, obj):
        if isinstance(obj, h5py.Dataset):
            return obj[()], None
        if isinstance(obj, h5py.Group):
            arr = None
            idx = None
            for sub in ("embedding", "embeddings", "features"):
                if sub in obj and isinstance(obj[sub], h5py.Dataset):
                    arr = obj[sub][()]
                    break
            if "indices" in obj and isinstance(obj["indices"], h5py.Dataset):
                idx = obj["indices"][()]
            if arr is not None:
                return arr, idx
        return None, None

    def _read_from_h5(self, pdb_id, chain_up):
        self._last_debug = {
            "h5_key": None,
            "h5_raw_len": None,
            "h5_after_zero_len": None,
            "h5_idx_raw_len": None,
        }
        if self.emb is None:
            return None, None
        if isinstance(self.emb, h5py.Group):
            for k in self._candidate_keys(pdb_id, chain_up):
                if k in self.emb:
                    arr, idx = self._read_group(self.emb[k])
                    if arr is None:
                        continue
                    self._last_debug["h5_key"] = k
                    self._last_debug["h5_raw_len"] = int(arr.shape[0])
                    if idx is not None:
                        self._last_debug["h5_idx_raw_len"] = int(np.asarray(idx).shape[0])
                    return arr, idx
            if self.fallback:
                for k in self.emb.keys():
                    arr, idx = self._read_group(self.emb[k])
                    if arr is None:
                        continue
                    self._last_debug["h5_key"] = k
                    self._last_debug["h5_raw_len"] = int(arr.shape[0])
                    if idx is not None:
                        self._last_debug["h5_idx_raw_len"] = int(np.asarray(idx).shape[0])
                    return arr, idx
        elif isinstance(self.emb, h5py.Dataset):
            arr = self.emb[()]
            self._last_debug["h5_key"] = "dataset"
            self._last_debug["h5_raw_len"] = int(arr.shape[0])
            return arr, None
        return None, None

    def get_last_debug_info(self):
        return dict(self._last_debug)

    @torch.no_grad()
    def encode_structure(self, pdb_path, chain_id, use_sequence=False):
        pdb_id = os.path.basename(pdb_path).split(".")[0].lower()
        chain_up = (chain_id or "").strip().upper()

        arr, idx = self._read_from_h5(pdb_id, chain_up)
        if arr is None:
            if self.skip_on_missing:
                raise MissingRepresentation(f"No H5 entry for {pdb_id}/{chain_up}")
            arr = np.zeros((1, self.embed_dim), dtype=np.float32)
            idx = None

        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.asarray(arr)
        if idx is not None:
            try:
                idx = np.asarray(idx).astype(int)
            except Exception:
                idx = None

        try:
            row_nonzero = ~np.all(arr == 0, axis=-1)
        except Exception:
            row_nonzero = np.ones((arr.shape[0],), dtype=bool)
        if idx is not None and idx.shape[0] == arr.shape[0]:
            mask = (idx != -1) & row_nonzero
        else:
            mask = row_nonzero

        arr = arr[mask]
        self._last_debug["h5_after_zero_len"] = int(arr.shape[0])
        if idx is not None and idx.shape[0] == mask.shape[0]:
            idx = idx[mask]
        elif idx is not None and idx.shape[0] == row_nonzero.shape[0]:
            idx = idx[mask]

        feats = torch.as_tensor(arr, dtype=self.output_dtype, device=self.device)
        L = int(feats.shape[0])
        if idx is None or idx.shape[0] != L:
            idx = np.arange(L, dtype=int)

        seqs = ["X"] * L if use_sequence else ["X"] * L
        return feats, idx, seqs
