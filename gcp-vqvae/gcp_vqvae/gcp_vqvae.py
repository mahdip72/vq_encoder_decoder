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

from ._internal.data.dataset import custom_collate_pretrained_gcp
from ._internal.demo.dataset import DemoStructureDataset
from ._internal.models.super_model import prepare_model
from ._internal.utils.utils import get_logger, load_checkpoints_simple, load_configs

DEFAULT_DROP_PREFIXES = ("protein_encoder.", "vqvae.decoder.esm_")
DEFAULT_SILENCED_LOGGERS = (
    "graphein",
    "Bio",
    "torch_geometric",
    "torch_cluster",
    "torch_scatter",
    "torch_sparse",
    "huggingface_hub",
    "transformers",
)
AA_TOKENS = "ARNDCQEGHILKMFPSTWYVX"


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


def _resolve_hf_path(
    model_id: str,
    filename: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required to download models from Hugging Face. "
            "Install it or provide local checkpoint/config paths."
        ) from exc

    path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    if logger is not None:
        logger.info("Resolved HF file %s from %s -> %s", filename, model_id, path)
    return path


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
    plddt_logits: Optional[torch.Tensor] = None,
    max_length: Optional[int] = None,
) -> List[dict]:
    records: List[dict] = []
    cpu_inds = indices_tensor.detach().cpu().tolist()
    if not isinstance(cpu_inds, list):
        cpu_inds = [cpu_inds]
    plddt_values = None
    if plddt_logits is not None:
        plddt_values = plddt_logits.squeeze(-1).detach().cpu().tolist()
        if not isinstance(plddt_values, list):
            plddt_values = [plddt_values]
        elif plddt_values and not isinstance(plddt_values[0], list):
            plddt_values = [plddt_values]
    for pid, idx, seq in zip(pids, cpu_inds, sequences):
        if not isinstance(idx, list):
            idx = [idx]
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]
        records.append({
            "pid": pid,
            "structures": idx[:len(seq)],
            "Amino Acid Sequence": seq,
            "plddt": None,
        })
    if plddt_values is not None:
        for record, values in zip(records, plddt_values):
            record["plddt"] = values[:len(record["structures"])]
    return records


def _stack_records(records: List[dict]) -> dict:
    if not records:
        return {}
    keys = records[0].keys()
    stacked = {key: [rec.get(key) for rec in records] for key in keys}
    return stacked


def _record_embeddings(
    pids: Sequence[str],
    embeddings: torch.Tensor,
    indices_tensor: torch.Tensor,
    sequences: Sequence[str],
    *,
    plddt_logits: Optional[torch.Tensor] = None,
    keep_missing_tokens: bool,
    max_length: Optional[int] = None,
) -> List[dict]:
    records: List[dict] = []
    emb_np = embeddings.detach().cpu().numpy()
    cpu_inds = indices_tensor.detach().cpu().tolist()
    plddt_values = None
    if plddt_logits is not None:
        plddt_values = plddt_logits.squeeze(-1).detach().cpu().numpy()

    if plddt_values is None:
        plddt_iter = [None] * len(pids)
    else:
        if plddt_values.ndim == 1:
            plddt_iter = [plddt_values]
        else:
            plddt_iter = plddt_values

    for pid, emb, ind_list, seq, plddt_row in zip(pids, emb_np, cpu_inds, sequences, plddt_iter):
        if max_length is not None and len(seq) > max_length:
            seq = seq[:max_length]
        emb_trim = emb[:len(seq)]
        ind_trim = ind_list[:len(seq)]
        plddt_trim = None
        if plddt_row is not None:
            plddt_trim = plddt_row[:len(seq)]
        if keep_missing_tokens:
            seq_out = seq
            ind_out = [int(v) for v in ind_trim]
            emb_out = emb_trim
            plddt_out = plddt_trim
        else:
            keep_positions = [i for i, v in enumerate(ind_trim) if v != -1]
            emb_out = emb_trim[keep_positions]
            ind_out = [int(ind_trim[i]) for i in keep_positions]
            seq_out = "".join(seq[i] for i in keep_positions)
            plddt_out = None
            if plddt_trim is not None:
                plddt_out = plddt_trim[keep_positions]

        records.append({
            "pid": pid,
            "embedding": emb_out.astype("float32", copy=False),
            "indices": ind_out,
            "protein_sequence": seq_out,
            "plddt": plddt_out,
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


def _extract_plddt(
    plddt_logits: Optional[torch.Tensor],
    masks: torch.Tensor,
) -> List[Optional[torch.Tensor]]:
    if plddt_logits is None:
        return [None] * masks.shape[0]
    values = plddt_logits.squeeze(-1).detach().cpu()
    if values.dim() == 1:
        values = values.unsqueeze(0)
    masks_cpu = masks.detach().cpu()
    outputs: List[Optional[torch.Tensor]] = []
    for row, mask in zip(values, masks_cpu):
        row = row.clone()
        row[~mask] = float("nan")
        outputs.append(row)
    return outputs


def _extract_aa_sequences(
    seq_logits: Optional[torch.Tensor],
    masks: torch.Tensor,
    sequence_lengths: Optional[torch.Tensor] = None,
) -> List[Optional[str]]:
    if seq_logits is None:
        return [None] * masks.shape[0]

    token_ids = seq_logits.argmax(dim=-1).detach().cpu()
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)

    if sequence_lengths is not None:
        lengths = sequence_lengths.to(torch.long).detach().cpu()
        positions = torch.arange(token_ids.shape[1]).unsqueeze(0)
        valid_masks = positions < lengths.unsqueeze(1)
    else:
        valid_masks = masks.detach().cpu()

    outputs: List[Optional[str]] = []
    for row_ids, row_mask in zip(token_ids, valid_masks):
        chars = [
            AA_TOKENS[int(idx)] if 0 <= int(idx) < len(AA_TOKENS) else "X"
            for idx, is_valid in zip(row_ids.tolist(), row_mask.tolist())
            if is_valid
        ]
        outputs.append("".join(chars))
    return outputs


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
        trained_model_dir: Optional[str] = None,
        checkpoint_path: str = "checkpoints/best_valid.pth",
        config_vqvae: str = "config_vqvae.yaml",
        config_encoder: str = "config_gcpnet_encoder.yaml",
        config_decoder: str = "config_geometric_decoder.yaml",
        hf_model_id: Optional[str] = "Mahdip72/gcp-vqvae-large",
        hf_revision: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_hf: Optional[bool] = None,
        mode: str = "encode",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        mixed_precision: str = "bf16",
        max_length: Optional[int] = None,
        max_task_samples: Optional[int] = None,
        use_pretrained_encoder: bool = False,
        deterministic: bool = False,
        seed: int = 0,
        suppress_logging: bool = True,
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
            hf_model_id: Hugging Face model repo ID used when fetching remote
                checkpoints/configs. Defaults to "Mahdip72/gcp-vqvae-large".
            hf_revision: Optional Hugging Face revision/branch/tag.
            hf_cache_dir: Optional cache directory for Hugging Face downloads.
            hf_token: Optional Hugging Face token for private repos.
            use_hf: Force using Hugging Face files when True. When None, Hugging
                Face is used if trained_model_dir is not provided and no absolute
                local paths are supplied.
            mode: Inference mode; one of "encode", "embed", "decode", or "all".
                Use "all" to load both encode/embed and decode paths in one object.
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
            suppress_logging: When True, set noisy third-party loggers to ERROR
                during inference setup.
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
            >>> model = GCPVQVAE(mode="encode")  # Uses Hugging Face default model.
            >>> model = GCPVQVAE(mode="all")  # Load encode/embed + decode paths.
        """
        if mode not in ("encode", "embed", "decode", "all"):
            raise ValueError("mode must be one of: encode, embed, decode, all")

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
        self.suppress_logging = bool(suppress_logging)

        self.logger = logger or get_logger("gcp_vqvae")
        if self.suppress_logging:
            self._silence_external_loggers()

        self.hf_model_id = hf_model_id
        self.hf_revision = hf_revision
        self.hf_cache_dir = hf_cache_dir
        self.hf_token = hf_token
        if use_hf is None:
            has_abs_paths = any(
                os.path.isabs(path)
                for path in (checkpoint_path, config_vqvae, config_encoder, config_decoder)
            )
            use_hf = trained_model_dir is None and not has_abs_paths and hf_model_id is not None
        self.use_hf = bool(use_hf)

        if self.use_hf:
            if not hf_model_id:
                raise ValueError("hf_model_id is required when use_hf is True.")
            self.trained_model_dir = None
            self.vqvae_config_path = _resolve_hf_path(
                hf_model_id,
                config_vqvae,
                revision=hf_revision,
                cache_dir=hf_cache_dir,
                token=hf_token,
                logger=self.logger,
            )
            self.encoder_config_path = _resolve_hf_path(
                hf_model_id,
                config_encoder,
                revision=hf_revision,
                cache_dir=hf_cache_dir,
                token=hf_token,
                logger=self.logger,
            )
            self.decoder_config_path = _resolve_hf_path(
                hf_model_id,
                config_decoder,
                revision=hf_revision,
                cache_dir=hf_cache_dir,
                token=hf_token,
                logger=self.logger,
            )
            self.checkpoint_path = _resolve_hf_path(
                hf_model_id,
                checkpoint_path,
                revision=hf_revision,
                cache_dir=hf_cache_dir,
                token=hf_token,
                logger=self.logger,
            )
        else:
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
        def _build_loaded_model(decoder_only_flag: bool):
            model = prepare_model(
                self.configs,
                self.logger,
                encoder_configs=self.encoder_configs,
                decoder_configs=self.decoder_configs,
                decoder_only=decoder_only_flag,
                encoder_config_path=self.encoder_config_path,
            )

            for param in model.parameters():
                param.requires_grad = False

            model.eval()
            model = load_checkpoints_simple(
                self.checkpoint_path,
                model,
                self.logger,
                decoder_only=decoder_only_flag,
                drop_prefixes=DEFAULT_DROP_PREFIXES,
            )
            model.to(self.device)
            return model

        self.model = None
        self.decoder_model = None
        if self.mode == "decode":
            self.model = _build_loaded_model(True)
        elif self.mode in ("encode", "embed"):
            self.model = _build_loaded_model(False)
        else:
            self.model = _build_loaded_model(False)
            self.decoder_model = _build_loaded_model(True)

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

    def _silence_external_loggers(self) -> None:
        """Reduce third-party logger verbosity during inference."""
        for name in DEFAULT_SILENCED_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)

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
        pdb_dir: Optional[str],
        max_task_samples: Optional[int],
        progress: bool,
    ):
        """Construct a dataset and collate function for PDB/CIF inputs."""
        if not pdb_dir:
            raise ValueError("pdb_dir is required.")

        if max_task_samples is not None:
            self.configs.train_settings.max_task_samples = int(max_task_samples)

        dataset = DemoStructureDataset(
            pdb_dir,
            max_length=self.max_length,
            encoder_config_path=self.encoder_config_path,
            max_task_samples=int(max_task_samples or 0),
            progress=progress,
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
        pdb_dir: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        max_task_samples: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[dict]:
        """Encode structures into VQ indices.

        Available when initialized with mode "encode" or "all".

        Args:
            pdb_dir: Directory of PDB/CIF files to preprocess and encode (required).
            batch_size: Batch size for inference.
            num_workers: DataLoader workers.
            shuffle: Shuffle dataset before encoding.
            max_task_samples: Optional override for number of samples to load.
            show_progress: Whether to show a progress bar (default: false).

        Returns:
            Dict with keys: pid, structures, Amino Acid Sequence, plddt.
        """
        if self.mode not in ("encode", "all"):
            raise RuntimeError("GCPVQVAE is not initialized in encode/all mode.")

        if self.deterministic:
            self._seed_everything()

        dataset, collate_fn = self._build_dataset(
            pdb_dir=pdb_dir,
            max_task_samples=max_task_samples,
            progress=show_progress,
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
            plddt_logits = output_dict.get("plddt_logits")
            records.extend(
                _record_indices(
                    batch["pid"],
                    indices,
                    batch["seq"],
                    plddt_logits=plddt_logits,
                    max_length=self.max_length,
                )
            )

        return _stack_records(records)

    def embed(
        self,
        *,
        pdb_dir: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        max_task_samples: Optional[int] = None,
        keep_missing_tokens: bool = False,
        show_progress: bool = False,
    ) -> List[dict]:
        """Embed structures into VQ embeddings and indices.

        Available when initialized with mode "embed" or "all".

        Args:
            pdb_dir: Directory of PDB/CIF files to preprocess and embed (required).
            batch_size: Batch size for inference.
            num_workers: DataLoader workers.
            shuffle: Shuffle dataset before embedding.
            max_task_samples: Optional override for number of samples to load.
            keep_missing_tokens: Keep -1 indices and corresponding embeddings.
            show_progress: Whether to show a progress bar (default: false).

        Returns:
            Dict with keys: pid, embedding, indices, protein_sequence, plddt.
        """
        if self.mode not in ("embed", "all"):
            raise RuntimeError("GCPVQVAE is not initialized in embed/all mode.")

        if self.deterministic:
            self._seed_everything()

        dataset, collate_fn = self._build_dataset(
            pdb_dir=pdb_dir,
            max_task_samples=max_task_samples,
            progress=show_progress,
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
            plddt_logits = output_dict.get("plddt_logits")

            records.extend(
                _record_embeddings(
                    batch["pid"],
                    embeddings,
                    indices,
                    batch["seq"],
                    plddt_logits=plddt_logits,
                    keep_missing_tokens=keep_missing_tokens,
                    max_length=self.max_length,
                )
            )

        return _stack_records(records)

    def decode(
        self,
        indices_batch: Sequence[Sequence[int] | str],
        *,
        pids: Optional[Sequence[str]] = None,
        batch_size: int = 1,
        show_progress: bool = False,
    ) -> dict:
        """Decode VQ indices into backbone coordinates.

        Available when initialized with mode "decode" or "all".

        Args:
            indices_batch: List of index sequences or space-delimited strings.
            pids: Optional identifiers for each sequence.
            batch_size: Batch size for decoding.
            show_progress: Whether to show a progress bar (default: false).

        Returns:
            Dict with keys: pid, coords, mask, plddt, AA.
        """
        if self.mode not in ("decode", "all"):
            raise RuntimeError("GCPVQVAE is not initialized in decode/all mode.")
        if not indices_batch:
            return []

        if self.deterministic:
            self._seed_everything()

        model = self.decoder_model or self.model
        if model is None:
            raise RuntimeError("Decoder model not initialized.")

        results: List[dict] = []
        if pids is None:
            pids = [f"sample_{idx}" for idx in range(len(indices_batch))]

        chunks = _chunked(list(indices_batch), batch_size)
        if show_progress:
            total_batches = (len(indices_batch) + batch_size - 1) // batch_size
            chunks = tqdm(chunks, total=total_batches, disable=not show_progress, leave=True)
            chunks.set_description("Decode")

        for batch_index, batch in enumerate(chunks):
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
                output_dict = model(payload, decoder_only=True)

            outputs = output_dict["outputs"].view(-1, self.max_length, 3, 3)
            outputs = outputs.detach().cpu()
            plddt_values = _extract_plddt(output_dict.get("plddt_logits"), mask_tensor)
            aa_values = _extract_aa_sequences(
                output_dict.get("seq_logits"),
                mask_tensor,
                output_dict.get("sequence_lengths"),
            )
            masks = mask_tensor.detach().cpu()

            for pid, coords, mask, plddt, aa in zip(batch_pids, outputs, masks, plddt_values, aa_values):
                results.append({
                    "pid": pid,
                    "coords": coords,
                    "mask": mask,
                    "plddt": plddt,
                    "AA": aa,
                })

        return _stack_records(results)
