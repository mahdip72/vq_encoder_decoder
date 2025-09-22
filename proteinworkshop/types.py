"""Minimal type aliases used by the trimmed ProteinWorkshop fork."""

from typing import Dict, Literal, NewType

import torch
from jaxtyping import Float

ActivationType = Literal[
    "relu",
    "elu",
    "leaky_relu",
    "tanh",
    "sigmoid",
    "none",
    "silu",
    "swish",
]

LossType = Literal["cross_entropy", "nll_loss", "mse_loss", "l1_loss", "dihedral_loss"]

EncoderOutput = NewType("EncoderOutput", Dict[str, torch.Tensor])
ModelOutput = NewType("ModelOutput", Dict[str, torch.Tensor])
Label = NewType("Label", Dict[str, torch.Tensor])

ScalarNodeFeature = Literal[
    "amino_acid_one_hot",
    "alpha",
    "kappa",
    "dihedrals",
    "sidechain_torsions",
    "sequence_positional_encoding",
]
VectorNodeFeature = Literal["orientation", "virtual_cb_vector"]
ScalarEdgeFeature = Literal["edge_distance", "sequence_distance"]
VectorEdgeFeature = Literal["edge_vectors", "pos_emb"]

OrientationTensor = NewType(
    "OrientationTensor", Float[torch.Tensor, "n_nodes 2 3"]
)
