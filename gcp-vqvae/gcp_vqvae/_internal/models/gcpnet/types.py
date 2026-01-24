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

EncoderOutput = NewType("EncoderOutput", Dict[str, torch.Tensor])

ScalarNodeFeature = Literal[
    "amino_acid_one_hot",
    "alpha",
    "kappa",
    "dihedrals",
    "sequence_positional_encoding",
]
VectorNodeFeature = Literal["orientation"]
ScalarEdgeFeature = Literal["edge_distance"]
VectorEdgeFeature = Literal["edge_vectors"]

OrientationTensor = NewType(
    "OrientationTensor", Float[torch.Tensor, "n_nodes 2 3"]
)
