import logging
import os
import random
from contextlib import nullcontext
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._internal.data.dataset import GCPNetDataset, custom_collate_pretrained_gcp
from ._internal.demo.dataset import DemoStructureDataset
from ._internal.models.super_model import prepare_model
from ._internal.utils.utils import get_logger, load_checkpoints_simple, load_configs

DEFAULT_DROP_PREFIXES = ("protein_encoder.", "vqvae.decoder.esm_")


def _load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as handle:
        data = yaml.full_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Malformed config file: {path}")
    return data


def _resolve_path(base_dir: Optional[str], path: str) -> str:
    if os.path.isabs(path):
        return path
    if base_dir is None:
        return path
    return os.path.join(base_dir, path)


def _load_encoder_decoder_configs(encoder_cfg_path: str, decoder_cfg_path: str) -> tuple[Box, Box]:
    with open(encoder_cfg_path) as handle:
        enc_cfg = yaml.full_load(handle)
    with open(decoder_cfg_path) as handle:
        dec_cfg = yaml.full_load(handle)
    return Box(enc_cfg), Box(dec_cfg)


def _record_indices(
    pids: Sequence[str],
    indices_tensor: torch.Tensor,
    sequences: Sequence[str],
    *,
    max_length: Optional[int] = None,
) -> List[dict]:
    records: List[dict] = []
    cpu_inds = indices_tensor.detach().cpu().tolist()
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    for pid, idx, seq in zip(pids, cpu_inds, sequences):
        if not isinstance(idx, list):
            idx = [idx]
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]
        records.append({
            "pid": pid,
            "structures": idx[:len(seq)],
            "Amino Acid Sequence": seq,
        })
    return records


def _record_embeddings(
    pids: Sequence[str],
    embeddings: torch.Tensor,
    indices_tensor: torch.Tensor,
    sequences: Sequence[str],
    *,
    keep_missing_tokens: bool,
    max_length: Optional[int] = None,
) -> List[dict]:
    records: List[dict] = []
    emb_np = embeddings.detach().cpu().numpy()
    cpu_inds = indices_tensor.detach().cpu().tolist()

    for pid, emb, ind_list, seq in zip(pids, emb_np, cpu_inds, sequences):
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]
        emb_trim = emb[:len(seq)]
        ind_trim = ind_list[:len(seq)]
        if keep_missing_tokens:
            seq_out = seq
            ind_out = [int(v) for v in ind_trim]
            emb_out = emb_trim
        else:
            keep_positions = [i for i, v in enumerate(ind_trim) if v != -1]
            emb_out = emb_trim[keep_positions]
            ind_out = [int(ind_trim[i]) for i in keep_positions]
            seq_out = "".join(seq[i] for i in keep_positions)

        records.append({
            "pid": pid,
            "embedding": emb_out.astype("float32", copy=False),
            "indices": ind_out,
            "protein_sequence": seq_out,
        })
    return records


def _prepare_decode_batch(
    indices_batch: Sequence[Sequence[int] | str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded = []
    masks = []

    for entry in indices_batch:
        if isinstance(entry, str):
            tokens = [tok for tok in entry.strip().split() if tok]
            indices = [int(tok) for tok in tokens]
        else:
            indices = [int(val) for val in entry]

        if len(indices) > max_length:
            indices = indices[:max_length]
        pad_len = max_length - len(indices)
        padded_indices = indices + [-1] * pad_len
        mask = [val != -1 for val in padded_indices]

        padded.append(padded_indices)
        masks.append(mask)

    indices_tensor = torch.tensor(padded, dtype=torch.long)
    mask_tensor = torch.tensor(masks, dtype=torch.bool)
    nan_mask = mask_tensor.clone()
    return indices_tensor, mask_tensor, nan_mask


def _chunked(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for idx in range(0, len(items), batch_size):
        yield items[idx:idx + batch_size]


def _autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


class GCPVQVAE:
    """GCP-VQVAE wrapper supporting encode, embed, and decode inference."""

    def __init__(
        self,
        *,
        trained_model_dir: Optional[str],
        checkpoint_path: str,
        config_vqvae: str,
        config_encoder: str,
        config_decoder: str,
        mode: str = "encode",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        mixed_precision: str = "bf16",
        max_length: Optional[int] = None,
        max_task_samples: Optional[int] = None,
        use_pretrained_encoder: bool = False,
        deterministic: bool = False,
        seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize a GCP-VQVAE inference wrapper.

        Args:
            trained_model_dir: Base directory for checkpoint/config paths. Use None
                if all paths are absolute.
            checkpoint_path: Path to the model checkpoint (.pth). Resolved relative
                to trained_model_dir when provided.
            config_vqvae: Path to the VQ-VAE config YAML. Resolved relative to
                trained_model_dir when provided.
            config_encoder: Path to the encoder config YAML. Resolved relative to
                trained_model_dir when provided.
            config_decoder: Path to the decoder config YAML. Resolved relative to
                trained_model_dir when provided.
            mode: Inference mode; one of "encode", "embed", or "decode".
            device: Torch device string (e.g., "cuda", "cpu"). Defaults to CUDA
                if available.
            dtype: Explicit torch dtype for autocast. If None, dtype is derived
                from mixed_precision.
            mixed_precision: Autocast policy when dtype is None. Use "bf16",
                "fp16", or "no" to disable. Defaults to "bf16" on CUDA.
            max_length: Optional override for configs.model.max_length.
            max_task_samples: Optional override for configs.train_settings.max_task_samples.
            use_pretrained_encoder: Enable pretrained encoder path if supported by config.
            deterministic: If True, enable deterministic kernels and seeding for
                repeatable results (may reduce performance).
            seed: Base seed used when deterministic is True.
            logger: Optional logger for checkpoint/config messages.

        Example:
            >>> model = GCPVQVAE(
            ...     trained_model_dir="/abs/path/to/trained_model",
            ...     checkpoint_path="checkpoints/best_valid.pth",
            ...     config_vqvae="config_vqvae.yaml",
            ...     config_encoder="config_gcpnet_encoder.yaml",
            ...     config_decoder="config_geometric_decoder.yaml",
            ...     mode="encode",
            ...     mixed_precision="bf16",
            ... )
        """
        if mode not in ("encode", "embed", "decode"):
            raise ValueError("mode must be one of: encode, embed, decode")

        self.mode = mode
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mixed_precision = mixed_precision
        self.dtype = dtype
        if self.dtype is None:
            self.dtype = self._resolve_mixed_precision_dtype()
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

        self.logger = logger or get_logger("gcp_vqvae")

        self.trained_model_dir = trained_model_dir
        self.vqvae_config_path = _resolve_path(trained_model_dir, config_vqvae)
        self.encoder_config_path = _resolve_path(trained_model_dir, config_encoder)
        self.decoder_config_path = _resolve_path(trained_model_dir, config_decoder)
        self.checkpoint_path = _resolve_path(trained_model_dir, checkpoint_path)

        vqvae_cfg = _load_yaml(self.vqvae_config_path)
        self.configs = load_configs(vqvae_cfg)

        if max_task_samples is not None:
            self.configs.train_settings.max_task_samples = int(max_task_samples)
        if max_length is not None:
            self.configs.model.max_length = int(max_length)

        esm_cfg = getattr(self.configs.train_settings.losses, "esm", None)
        if esm_cfg and getattr(esm_cfg, "enabled", False):
            esm_cfg.enabled = False
        self.configs.model.encoder.pretrained.enabled = bool(use_pretrained_encoder)

        self.encoder_configs, self.decoder_configs = _load_encoder_decoder_configs(
            self.encoder_config_path,
            self.decoder_config_path,
        )

        if self.deterministic:
            self._apply_deterministic_settings()

        decoder_only = self.mode == "decode"
        self.model = prepare_model(
            self.configs,
            self.logger,
            encoder_configs=self.encoder_configs,
            decoder_configs=self.decoder_configs,
            decoder_only=decoder_only,
            encoder_config_path=self.encoder_config_path,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model = load_checkpoints_simple(
            self.checkpoint_path,
            self.model,
            self.logger,
            decoder_only=decoder_only,
            drop_prefixes=DEFAULT_DROP_PREFIXES,
        )
        self.model.to(self.device)

    def _resolve_mixed_precision_dtype(self) -> Optional[torch.dtype]:
        """Resolve mixed_precision into a torch dtype for autocast."""
        if self.device.type != "cuda":
            return None
        value = str(self.mixed_precision).lower()
        if value in ("no", "none", "false", "0") or not value:
            return None
        if value in ("bf16", "bfloat16"):
            return torch.bfloat16
        if value in ("fp16", "float16"):
            return torch.float16
        raise ValueError(f"Unsupported mixed_precision value: {self.mixed_precision}")

    def _apply_deterministic_settings(self) -> None:
        """Enable deterministic CUDA behavior for repeatable inference."""
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

    def _seed_everything(self) -> None:
        """Seed Python, NumPy, and torch RNGs."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _dataloader_seed_kwargs(self) -> dict:
        """Return DataLoader seeding kwargs when deterministic is enabled."""
        if not self.deterministic:
            return {}

        def _seed_worker(worker_id: int) -> None:
            worker_seed = self.seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return {"worker_init_fn": _seed_worker, "generator": generator}

    @property
    def max_length(self) -> int:
        """Maximum sequence length used for padding/truncation."""
        return int(self.configs.model.max_length)

    def _build_dataset(
        self,
        *,
        data_path: Optional[str],
        pdb_dir: Optional[str],
        max_task_samples: Optional[int],
    ):
        """Construct a dataset and collate function for H5 or PDB/CIF inputs."""
        if data_path and pdb_dir:
            raise ValueError("Provide only one of data_path or pdb_dir.")
        if not data_path and not pdb_dir:
            raise ValueError("Either data_path or pdb_dir is required.")

        if max_task_samples is not None:
            self.configs.train_settings.max_task_samples = int(max_task_samples)

        if data_path:
            dataset = GCPNetDataset(
                data_path,
                top_k=self.encoder_configs.top_k,
                num_positional_embeddings=self.encoder_configs.num_positional_embeddings,
                configs=self.configs,
                mode="evaluation",
                esm_tokenizer=None,
                encoder_config_path=self.encoder_config_path,
            )
        else:
            dataset = DemoStructureDataset(
                pdb_dir,
                max_length=self.max_length,
                encoder_config_path=self.encoder_config_path,
                max_task_samples=int(max_task_samples or 0),
                progress=False,
            )

        collate_fn = lambda batch: custom_collate_pretrained_gcp(
            batch,
            featuriser=dataset.pretrained_featuriser,
            task_transform=dataset.pretrained_task_transform,
        )

        return dataset, collate_fn

    def encode(
        self,
        *,
        data_path: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        max_task_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[dict]:
        """Encode structures into VQ indices.

        Args:
            data_path: Directory of H5 files to encode.
            pdb_dir: Directory of PDB/CIF files to preprocess and encode.
            batch_size: Batch size for inference.
            num_workers: DataLoader workers.
            shuffle: Shuffle dataset before encoding.
            max_task_samples: Optional override for number of samples to load.
            show_progress: Whether to show a progress bar.

        Returns:
            List of dicts with keys: pid, structures, Amino Acid Sequence.
        """
        if self.mode != "encode":
            raise RuntimeError("GCPVQVAE is not initialized in encode mode.")

        if self.deterministic:
            self._seed_everything()

        dataset, collate_fn = self._build_dataset(
            data_path=data_path,
            pdb_dir=pdb_dir,
            max_task_samples=max_task_samples,
        )

        loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=self.device.type == "cuda",
            **self._dataloader_seed_kwargs(),
        )

        records: List[dict] = []
        iterator = tqdm(loader, total=len(loader), disable=not show_progress, leave=True)
        iterator.set_description("Encode")

        for batch in iterator:
            batch["graph"] = batch["graph"].to(self.device)
            batch["masks"] = batch["masks"].to(self.device)
            batch["nan_masks"] = batch["nan_masks"].to(self.device)

            with torch.inference_mode(), _autocast_context(self.device, self.dtype):
                output_dict = self.model(batch, return_vq_layer=True)

            indices = output_dict["indices"]
            records.extend(
                _record_indices(
                    batch["pid"],
                    indices,
                    batch["seq"],
                    max_length=self.max_length,
                )
            )

        return records

    def embed(
        self,
        *,
        data_path: Optional[str] = None,
        pdb_dir: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        max_task_samples: Optional[int] = None,
        keep_missing_tokens: bool = False,
        show_progress: bool = True,
    ) -> List[dict]:
        """Embed structures into VQ embeddings and indices.

        Args:
            data_path: Directory of H5 files to embed.
            pdb_dir: Directory of PDB/CIF files to preprocess and embed.
            batch_size: Batch size for inference.
            num_workers: DataLoader workers.
            shuffle: Shuffle dataset before embedding.
            max_task_samples: Optional override for number of samples to load.
            keep_missing_tokens: Keep -1 indices and corresponding embeddings.
            show_progress: Whether to show a progress bar.

        Returns:
            List of dicts with keys: pid, embedding, indices, protein_sequence.
        """
        if self.mode != "embed":
            raise RuntimeError("GCPVQVAE is not initialized in embed mode.")

        if self.deterministic:
            self._seed_everything()

        dataset, collate_fn = self._build_dataset(
            data_path=data_path,
            pdb_dir=pdb_dir,
            max_task_samples=max_task_samples,
        )

        loader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=self.device.type == "cuda",
            **self._dataloader_seed_kwargs(),
        )

        records: List[dict] = []
        iterator = tqdm(loader, total=len(loader), disable=not show_progress, leave=True)
        iterator.set_description("Embed")

        for batch in iterator:
            batch["graph"] = batch["graph"].to(self.device)
            batch["masks"] = batch["masks"].to(self.device)
            batch["nan_masks"] = batch["nan_masks"].to(self.device)

            with torch.inference_mode(), _autocast_context(self.device, self.dtype):
                output_dict = self.model(batch, return_vq_layer=True)

            embeddings = output_dict["embeddings"]
            indices = output_dict["indices"]

            records.extend(
                _record_embeddings(
                    batch["pid"],
                    embeddings,
                    indices,
                    batch["seq"],
                    keep_missing_tokens=keep_missing_tokens,
                    max_length=self.max_length,
                )
            )

        return records

    def decode(
        self,
        indices_batch: Sequence[Sequence[int] | str],
        *,
        pids: Optional[Sequence[str]] = None,
        batch_size: int = 1,
    ) -> List[dict]:
        """Decode VQ indices into backbone coordinates.

        Args:
            indices_batch: List of index sequences or space-delimited strings.
            pids: Optional identifiers for each sequence.
            batch_size: Batch size for decoding.

        Returns:
            List of dicts with keys: pid, coords, mask.
        """
        if self.mode != "decode":
            raise RuntimeError("GCPVQVAE is not initialized in decode mode.")
        if not indices_batch:
            return []

        if self.deterministic:
            self._seed_everything()

        results: List[dict] = []
        if pids is None:
            pids = [f"sample_{idx}" for idx in range(len(indices_batch))]

        for batch_index, batch in enumerate(_chunked(list(indices_batch), batch_size)):
            indices_tensor, mask_tensor, nan_mask = _prepare_decode_batch(batch, self.max_length)
            batch_pids = pids[batch_index * batch_size: batch_index * batch_size + len(batch)]

            payload = {
                "indices": indices_tensor.to(self.device),
                "masks": mask_tensor.to(self.device),
                "nan_masks": nan_mask.to(self.device),
                "pid": list(batch_pids),
                "seq": ["" for _ in batch_pids],
            }

            with torch.inference_mode(), _autocast_context(self.device, self.dtype):
                output_dict = self.model(payload, decoder_only=True)

            outputs = output_dict["outputs"].view(-1, self.max_length, 3, 3)
            outputs = outputs.detach().cpu()
            masks = mask_tensor.detach().cpu()

            for pid, coords, mask in zip(batch_pids, outputs, masks):
                results.append({
                    "pid": pid,
                    "coords": coords,
                    "mask": mask,
                })

        return results
