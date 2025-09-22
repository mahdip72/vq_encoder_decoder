from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn
import yaml
from torch_geometric.data import Batch

from graphein.protein.tensor.data import ProteinBatch
from proteinworkshop.types import EncoderOutput


ModuleSpec = Mapping[str, Any]


def _import_from_string(path: str) -> Any:
    if "." not in path:
        raise ValueError(f"Invalid import path '{path}'. Expected 'module.Class'.")
    module_path, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Module '{module_path}' has no attribute '{attr}'.") from exc


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: _to_namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(item) for item in obj]
    return obj


def instantiate_module(spec: Optional[ModuleSpec]) -> Optional[nn.Module]:
    if not spec:
        return None
    target = spec.get("module")
    if target is None:
        raise ValueError("Module specification must include a 'module' entry.")
    constructor = _import_from_string(target)
    kwargs = spec.get("kwargs", {})
    return constructor(**kwargs)


def load_pretrained_config(path: Union[str, bytes]) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, Mapping):
        raise TypeError("Pretrained configuration must be a mapping.")
    return cfg


@dataclass
class PretrainedComponents:
    featuriser: nn.Module
    encoder: nn.Module
    task_transform: Optional[nn.Module]


class PretrainedEncoder(nn.Module):
    """Minimal wrapper that mirrors the old Lightning BenchMarkModel interface."""

    def __init__(self, components: PretrainedComponents) -> None:
        super().__init__()
        self.featuriser = components.featuriser
        self.encoder = components.encoder
        self.task_transform = components.task_transform

    def featurise(self, batch: Union[Batch, ProteinBatch]) -> Union[Batch, ProteinBatch]:
        batch = self.featuriser(batch)
        if self.task_transform is not None:
            batch = self.task_transform(batch)
        return batch

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        batch = self.featurise(batch)
        return self.encoder(batch)


def _build_components(cfg: Mapping[str, Any]) -> PretrainedComponents:
    feature_spec = cfg.get("features")
    if feature_spec is None:
        raise KeyError("Pretrained configuration must define 'features'.")
    featuriser = instantiate_module(feature_spec)

    task_cfg = cfg.get("task", {})
    task_transform = instantiate_module(task_cfg.get("transform")) if isinstance(task_cfg, Mapping) else None

    encoder_spec = cfg.get("encoder")
    if encoder_spec is None:
        raise KeyError("Pretrained configuration must define 'encoder'.")
    encoder_kwargs = dict(encoder_spec.get("kwargs", {}))
    for key in ("module_cfg", "model_cfg", "layer_cfg"):
        if key in encoder_kwargs and isinstance(encoder_kwargs[key], dict):
            encoder_kwargs[key] = _to_namespace(encoder_kwargs[key])

    constructor = _import_from_string(encoder_spec["module"])
    encoder = constructor(**encoder_kwargs)

    return PretrainedComponents(
        featuriser=featuriser,
        encoder=encoder,
        task_transform=task_transform,
    )


def _extract_prefixed(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Mapping[str, torch.Tensor]:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _load_checkpoint(path: str, map_location: Union[str, torch.device]) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_pretrained_encoder(
    config_path: str,
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
) -> PretrainedEncoder:
    cfg = load_pretrained_config(config_path)
    components = _build_components(cfg)

    state_dict = _load_checkpoint(checkpoint_path, map_location=map_location)
    encoder_state = _extract_prefixed(state_dict, "encoder.")
    featuriser_state = _extract_prefixed(state_dict, "featuriser.")
    transform_state = _extract_prefixed(state_dict, "task_transform.")

    if featuriser_state:
        components.featuriser.load_state_dict(featuriser_state, strict=strict)
    if transform_state and components.task_transform is not None:
        components.task_transform.load_state_dict(transform_state, strict=strict)
    if encoder_state:
        components.encoder.load_state_dict(encoder_state, strict=strict)

    return PretrainedEncoder(components)


__all__ = [
    "PretrainedEncoder",
    "PretrainedComponents",
    "instantiate_module",
    "load_pretrained_config",
    "load_pretrained_encoder",
]
