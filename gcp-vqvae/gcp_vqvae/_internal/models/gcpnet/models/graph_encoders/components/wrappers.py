from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from jaxtyping import Bool

from gcp_vqvae._internal.models.gcpnet.typecheck import jaxtyped, typechecker


@dataclass(frozen=True, slots=True)
class ScalarVector:
    """Lightweight container for paired scalar/vector features."""

    scalar: torch.Tensor
    vector: torch.Tensor

    # tuple-like helpers -------------------------------------------------
    def __iter__(self):
        yield self.scalar
        yield self.vector

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self.scalar
        if index == 1:
            return self.vector
        raise IndexError("ScalarVector only contains two elements")

    # elementwise arithmetic --------------------------------------------
    def __add__(self, other: "ScalarVector") -> "ScalarVector":
        if not isinstance(other, ScalarVector):  # pragma: no cover - defensive
            return NotImplemented
        return ScalarVector(self.scalar + other.scalar, self.vector + other.vector)

    def __mul__(self, other) -> "ScalarVector":
        if isinstance(other, ScalarVector):
            return ScalarVector(self.scalar * other.scalar, self.vector * other.vector)
        return ScalarVector(self.scalar * other, self.vector * other)

    # utility helpers ----------------------------------------------------
    def concat(self, others: Iterable["ScalarVector"], dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        dim %= self.scalar.dim()
        scalars = [self.scalar]
        vectors = [self.vector]
        for other in others:
            if not isinstance(other, ScalarVector):  # pragma: no cover - defensive
                raise TypeError("Expected ScalarVector instances in concat")
            scalars.append(other.scalar)
            vectors.append(other.vector)
        return torch.cat(scalars, dim=dim), torch.cat(vectors, dim=dim)

    def flatten(self) -> torch.Tensor:
        flat_vector = self.vector.reshape(*self.vector.shape[:-2], -1)
        return torch.cat((self.scalar, flat_vector), dim=-1)

    @staticmethod
    def recover(x: torch.Tensor, vector_dim: int) -> "ScalarVector":
        if vector_dim == 0:
            zero_vector = x.new_zeros(*x.shape[:-1], 0, 3)
            return ScalarVector(x, zero_vector)
        v = x[..., -3 * vector_dim :].reshape(*x.shape[:-1], vector_dim, 3)
        s = x[..., : -3 * vector_dim]
        return ScalarVector(s, v)

    def idx(self, index) -> "ScalarVector":
        return ScalarVector(self.scalar[index], self.vector[index])

    def clone(self) -> "ScalarVector":
        return ScalarVector(self.scalar.clone(), self.vector.clone())

    @jaxtyped(typechecker=typechecker)
    def mask(self, node_mask: Bool[torch.Tensor, " n_nodes"]) -> "ScalarVector":
        return ScalarVector(
            self.scalar * node_mask[..., None],
            self.vector * node_mask[..., None, None],
        )


__all__ = ["ScalarVector"]
