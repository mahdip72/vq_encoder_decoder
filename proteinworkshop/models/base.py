from __future__ import annotations

from typing import Optional, Union

import hydra
import lightning as L
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Batch

from graphein.protein.tensor.data import ProteinBatch
from proteinworkshop.types import EncoderOutput


class BaseModel(L.LightningModule):
    """Minimal Lightning module used for loading pretrained checkpoints only."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.config = cfg
        self.featuriser: nn.Module = hydra.utils.instantiate(cfg.features)
        self.task_transform = hydra.utils.instantiate(cfg.get("task.transform"))

    # ---------------------------------------------------------------------
    # Lightning API (stubbed for evaluation-only usage)
    # ---------------------------------------------------------------------
    def training_step(self, *_args, **_kwargs):  # pragma: no cover - safety guard
        raise RuntimeError("Training is disabled in the trimmed ProteinWorkshop fork.")

    def validation_step(self, *_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Validation is disabled in the trimmed ProteinWorkshop fork.")

    def test_step(self, *_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("Testing is disabled in the trimmed ProteinWorkshop fork.")

    def configure_optimizers(self):  # pragma: no cover
        raise RuntimeError("Optimizers are not configured in the trimmed ProteinWorkshop fork.")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def featurise(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Union[Batch, ProteinBatch]:
        out = self.featuriser(batch)
        if self.task_transform is not None:
            out = self.task_transform(out)
        return out

    def transform_encoder_output(
        self, output: EncoderOutput, _batch: Union[Batch, ProteinBatch]
    ) -> EncoderOutput:
        return output

    def compute_output(
        self, output: EncoderOutput, _batch: Union[Batch, ProteinBatch]
    ) -> EncoderOutput:
        return output


class BenchMarkModel(BaseModel):
    """Lean wrapper that exposes the pretrained encoder for inference."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.encoder: nn.Module = hydra.utils.instantiate(cfg.encoder)
        self.decoder: Optional[nn.Module] = None
        self.losses = {}
        self.metric_names: list[str] = []

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        output: EncoderOutput = self.encoder(batch)
        output = self.transform_encoder_output(output, batch)
        return self.compute_output(output, batch)
