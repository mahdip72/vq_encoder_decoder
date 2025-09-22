from typing import Literal

from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from torch_geometric.data import Batch


@jaxtyped(typechecker=typechecker)
def transform_representation(
    batch: Batch, representation_type: Literal["CA"]
) -> Batch:
    """Assign ``batch.pos`` to the Cα coordinates.

    The trimmed inference stack only supports single-residue (Cα) graphs; any
    other representation request is treated as a configuration error.
    """

    if representation_type != "CA":  # pragma: no cover - defensive guard
        raise ValueError(
            "Trimmed ProteinWorkshop only supports the 'CA' representation"
        )

    batch.pos = batch.coords[:, 1, :]
    return batch


__all__ = ["transform_representation"]
