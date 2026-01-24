###########################################################################################
# Implementation of Geometric-Complete Perceptron layers
#
# Papers:
# (1) Geometry-Complete Perceptron Networks for 3D Molecular Graphs,
#     by A Morehead, J Cheng
# (2) Geometry-Complete Diffusion for 3D Molecule Generation,
#     by A Morehead, J Cheng
#
# Orginal repositories:
# (1) https://github.com/BioinfoMachineLearning/GCPNet
# (2) https://github.com/BioinfoMachineLearning/Bio-Diffusion
###########################################################################################

from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union
import contextlib

import torch
import torch._dynamo as dynamo
import torch_scatter

_HAS_CUGRAPH_OPS = False

try:  # Optional fused message-passing dependency
    import cupy as cp  # type: ignore
    from pylibcugraphops import make_csc  # type: ignore
    from pylibcugraphops import operators as cgo  # type: ignore

    _HAS_CUGRAPH_OPS = True
except ImportError:  # pragma: no cover - fallback when cuGraph-ops is unavailable
    cp = None  # type: ignore
    make_csc = None  # type: ignore
    cgo = None  # type: ignore


@dataclass
class _CuGraphCSCCache:
    perm: torch.Tensor
    graph: Any
    num_edges: int
    num_nodes: int
    offsets_cp: Any
    indices_cp: Any

from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import Bool, Float, Int64
from torch import nn
from torch_geometric.data import Batch

from ..components import radial
from ..components.wrappers import (
    ScalarVector,
)
from ...utils import (
    get_activations,
    is_identity,
    safe_norm,
)
from gcp_vqvae._internal.models.gcpnet.typecheck import jaxtyped, typechecker

try:  # Optional dependency for backwards compatibility
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when OmegaConf is unavailable
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore


@dynamo.disable
def _torch_to_cupy(tensor: torch.Tensor):
    if cp is None:
        raise RuntimeError("CuPy is required for cuGraph-ops aggregation.")
    return cp.asarray(tensor)


@dynamo.disable
def _build_cugraph_cache(
    row: torch.Tensor, col: torch.Tensor, num_nodes: int
) -> _CuGraphCSCCache:
    if cp is None or make_csc is None:
        raise RuntimeError("pylibcugraphops is required for cuGraph-ops aggregation.")
    if row.numel() == 0:
        empty = row.new_empty(0, dtype=torch.long, device=row.device)
        return _CuGraphCSCCache(
            perm=empty,
            graph=None,
            num_edges=0,
            num_nodes=num_nodes,
            offsets_cp=None,
            indices_cp=None,
        )

    dest = row.to(torch.int64)
    src = col.to(torch.int64)

    sorted_dest, perm = torch.sort(dest)
    src_sorted = src.index_select(0, perm)

    counts = torch.bincount(sorted_dest, minlength=num_nodes)
    offsets = torch.empty(num_nodes + 1, device=row.device, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = counts.cumsum(0)

    offsets_cp = _torch_to_cupy(offsets)
    indices_cp = _torch_to_cupy(src_sorted)

    max_src = int(src.max().item()) + 1 if src.numel() else 0
    num_src_nodes = max(num_nodes, max_src)
    graph = make_csc(offsets_cp, indices_cp, num_src_nodes)

    return _CuGraphCSCCache(
        perm=perm,
        graph=graph,
        num_edges=row.numel(),
        num_nodes=num_nodes,
        offsets_cp=offsets_cp,
        indices_cp=indices_cp,
    )


class _CuGraphSimpleE2N(torch.autograd.Function):
    """Edge-to-node aggregation backed by pylibcugraphops."""

    @staticmethod
    @dynamo.disable
    def forward(
        ctx,
        messages: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        num_nodes: int,
        cache: Optional[_CuGraphCSCCache] = None,
    ) -> torch.Tensor:
        if cp is None or make_csc is None or cgo is None:
            raise RuntimeError("pylibcugraphops is not available but CuGraph aggregation was requested.")
        if messages.numel() == 0:
            ctx.save_for_backward(row.new_empty(0))
            ctx.graph = None
            ctx.num_edges = 0
            ctx.feature_dim = messages.shape[-1]
            return messages.new_zeros(num_nodes, messages.shape[-1])
        if cache is None:
            cache = _build_cugraph_cache(row, col, num_nodes)

        if cache.graph is None and messages.numel() != 0:
            raise RuntimeError("Cached cuGraph graph is missing for non-empty edge set.")
        if cache.num_edges != messages.shape[0]:
            raise RuntimeError("Cached cuGraph graph has mismatched number of edges.")

        perm = cache.perm
        messages_sorted = messages.index_select(0, perm)
        feature_dim = messages_sorted.shape[-1]

        # Convert to float32 for the fused CUDA kernel if necessary
        messages_comp = messages_sorted.to(torch.float32)
        output_comp = torch.zeros(num_nodes, feature_dim, device=messages.device, dtype=torch.float32)

        cgo.agg_simple_e2n_fwd(
            _torch_to_cupy(output_comp),
            _torch_to_cupy(messages_comp),
            cache.graph,
        )

        output = output_comp.to(messages.dtype)

        ctx.save_for_backward(perm)
        ctx.graph = cache.graph
        ctx.num_edges = messages.shape[0]
        ctx.feature_dim = feature_dim
        ctx.messages_dtype = messages.dtype
        return output

    @staticmethod
    @dynamo.disable
    def backward(ctx, grad_output: torch.Tensor):
        (perm,) = ctx.saved_tensors

        if ctx.num_edges == 0:
            grad_messages = grad_output.new_zeros((0, grad_output.shape[-1]))
            return grad_messages, None, None, None, None

        grad_output_comp = grad_output.to(torch.float32)
        grad_input_comp = torch.zeros(ctx.num_edges, ctx.feature_dim, device=grad_output.device, dtype=torch.float32)

        cgo.agg_simple_e2n_bwd(_torch_to_cupy(grad_input_comp), _torch_to_cupy(grad_output_comp), ctx.graph)

        grad_messages_sorted = grad_input_comp.to(grad_output.dtype)
        grad_messages = torch.empty_like(grad_messages_sorted)
        grad_messages[perm] = grad_messages_sorted

        return grad_messages, None, None, None, None


def _cfg_to_dict(cfg: Any) -> Any:
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        return copy(OmegaConf.to_container(cfg, throw_on_missing=True))
    if isinstance(cfg, dict):
        return {key: _cfg_to_dict(value) for key, value in cfg.items()}
    if hasattr(cfg, "__dict__"):
        return {key: _cfg_to_dict(value) for key, value in vars(cfg).items()}
    if isinstance(cfg, list):
        return [_cfg_to_dict(item) for item in cfg]
    return cfg


class VectorDropout(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class GCPDropout(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float, use_gcp_dropout: bool = True):
        super().__init__()
        self.scalar_dropout = (
            nn.Dropout(drop_rate) if use_gcp_dropout else nn.Identity()
        )
        self.vector_dropout = (
            VectorDropout(drop_rate) if use_gcp_dropout else nn.Identity()
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Union[torch.Tensor, ScalarVector]
    ) -> Union[torch.Tensor, ScalarVector]:
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (
            x.scalar.shape[0] == 0 or x.vector.shape[0] == 0
        ):
            return x
        elif isinstance(x, torch.Tensor):
            return self.scalar_dropout(x)
        return ScalarVector(self.scalar_dropout(x[0]), self.vector_dropout(x[1]))


class GCPLayerNorm(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(
        self, dims: ScalarVector, eps: float = 1e-8, use_gcp_norm: bool = True
    ):
        super().__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = (
            nn.LayerNorm(self.scalar_dims) if use_gcp_norm else nn.Identity()
        )
        self.use_gcp_norm = use_gcp_norm
        self.eps = eps

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def norm_vector(
        v: torch.Tensor, use_gcp_norm: bool = True, eps: float = 1e-8
    ) -> torch.Tensor:
        v_norm = v
        if use_gcp_norm:
            vector_norm = torch.clamp(
                torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps
            )
            vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
            v_norm = v / vector_norm
        return v_norm

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Union[torch.Tensor, ScalarVector]
    ) -> Union[torch.Tensor, ScalarVector]:
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (
            x.scalar.shape[0] == 0 or x.vector.shape[0] == 0
        ):
            return x
        elif not self.vector_dims:
            return self.scalar_norm(x)
        s, v = x
        return ScalarVector(
            self.scalar_norm(s),
            self.norm_vector(v, use_gcp_norm=self.use_gcp_norm, eps=self.eps),
        )


class GCP(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
        scalar_out_nonlinearity: Optional[str] = "silu",
        scalar_gate: int = 0,
        vector_gate: bool = True,
        feedforward_out: bool = False,
        bottleneck: int = 1,
        scalarization_vectorization_output_dim: int = 3,
        enable_e3_equivariance: bool = False,
        **kwargs,
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = ("none", "none")

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_activations(nonlinearities[0], return_functional=True),
            get_activations(nonlinearities[1], return_functional=True),
        )
        self.scalar_gate, self.vector_gate = scalar_gate, vector_gate
        self.enable_e3_equivariance = enable_e3_equivariance

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = (
                self.vector_input_dim // bottleneck
                if bottleneck > 1
                else max(self.vector_input_dim, self.vector_output_dim)
            )

            scalar_vector_frame_dim = scalarization_vectorization_output_dim * 3
            self.vector_down = nn.Linear(
                self.vector_input_dim, self.hidden_dim, bias=False
            )
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(
                        self.hidden_dim
                        + self.scalar_input_dim
                        + scalar_vector_frame_dim,
                        self.scalar_output_dim,
                    ),
                    get_activations(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(
                    self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim,
                    self.scalar_output_dim,
                )
            )

            self.vector_down_frames = nn.Linear(
                self.vector_input_dim,
                scalarization_vectorization_output_dim,
                bias=False,
            )

            if self.vector_output_dim:
                self.vector_up = nn.Linear(
                    self.hidden_dim, self.vector_output_dim, bias=False
                )
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(
                        self.scalar_output_dim, self.vector_output_dim
                    )
        else:
            self.scalar_out = (
                nn.Sequential(
                    nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                    get_activations(scalar_out_nonlinearity),
                    nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                )
                if feedforward_out
                else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)
            )

    @jaxtyped(typechecker=typechecker)
    def create_zero_vector(
        self,
        scalar_rep: Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
    ) -> Float[torch.Tensor, "batch_num_entities o 3"]:
        return torch.zeros(
            scalar_rep.shape[0],
            self.vector_output_dim,
            3,
            device=scalar_rep.device,
        )

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def scalarize(
        vector_rep: Float[torch.Tensor, "batch_num_entities 3 3"],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_inputs: bool,
        enable_e3_equivariance: bool,
        dim_size: int,
        node_mask: Optional[Bool[torch.Tensor, "n_nodes"]] = None,
    ) -> Float[torch.Tensor, "effective_batch_num_entities 9"]:
        row, col = edge_index[0], edge_index[1]

        # gather source node features for each `entity` (i.e., node or edge)
        # note: edge inputs are already ordered according to source nodes
        vector_rep_i = vector_rep[row] if node_inputs else vector_rep

        # project equivariant values onto corresponding local frames
        if vector_rep_i.ndim == 2:
            vector_rep_i = vector_rep_i.unsqueeze(-1)
        elif vector_rep_i.ndim == 3:
            vector_rep_i = vector_rep_i.transpose(-1, -2)

        if node_mask is not None:
            edge_mask = node_mask[row] & node_mask[col]
            mask = edge_mask.to(frames.dtype).view(-1, 1, 1)
            matmul_result = torch.matmul(frames, vector_rep_i)
            matmul_result = matmul_result * mask
        else:
            matmul_result = torch.matmul(frames, vector_rep_i)

        local_scalar_rep_i = matmul_result.transpose(-1, -2)

        # potentially enable E(3)-equivariance and, thereby, chirality-invariance
        if enable_e3_equivariance:
            # avoid corrupting gradients with an in-place operation
            local_scalar_rep_i_copy = local_scalar_rep_i.clone()
            local_scalar_rep_i_copy[:, :, 1] = torch.abs(local_scalar_rep_i[:, :, 1])
            local_scalar_rep_i = local_scalar_rep_i_copy

        # reshape frame-derived geometric scalars
        local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

        if node_inputs:
            # for node inputs, summarize all edge-wise geometric scalars using an average
            return torch_scatter.scatter(
                local_scalar_rep_i,
                # summarize according to source node indices due to the directional nature of GCP's equivariant frames
                row,
                dim=0,
                dim_size=dim_size,
                reduce="mean",
            )

        return local_scalar_rep_i

    @jaxtyped(typechecker=typechecker)
    def vectorize(
        self,
        scalar_rep: Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
        vector_hidden_rep: Float[torch.Tensor, "batch_num_entities 3 n"],
    ) -> Float[torch.Tensor, "batch_num_entities o 3"]:
        vector_rep = self.vector_up(vector_hidden_rep)
        vector_rep = vector_rep.transpose(-1, -2)

        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(
                safe_norm(vector_rep, dim=-1, keepdim=True)
            )

        return vector_rep

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                Float[torch.Tensor, "batch_num_entities scalar_dim"],
                Float[torch.Tensor, "batch_num_entities m vector_dim"],
            ],
            Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
        ],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_inputs: bool = False,
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Union[
        Tuple[
            Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
            Float[torch.Tensor, "batch_num_entities n vector_dim"],
        ],
        Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            # curate direction-robust and (by default) chirality-aware scalar geometric features
            vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
            scalar_hidden_rep = self.scalarize(
                vector_down_frames_hidden_rep.transpose(-1, -2),
                edge_index,
                frames,
                node_inputs=node_inputs,
                enable_e3_equivariance=self.enable_e3_equivariance,
                dim_size=vector_down_frames_hidden_rep.shape[0],
                node_mask=node_mask,
            )
            merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        else:
            # update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.vectorize(scalar_rep, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        return ScalarVector(scalar_rep, vector_rep)


class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: ScalarVector,
        node_input_dims: ScalarVector,
        edge_hidden_dims: ScalarVector,
        node_hidden_dims: ScalarVector,
        num_atom_types: int = 0,
        nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
        cfg: Any = None,
        pre_norm: bool = True,
        use_gcp_norm: bool = True,
    ):
        super().__init__()

        if num_atom_types > 0:
            self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        else:
            self.atom_embedding = None

        self.radial_embedding = radial.CachedGaussianRBF(
            max_distance=cfg.r_max,
            num_rbf=cfg.num_rbf,
        )

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_normalization = GCPLayerNorm(
                edge_input_dims, use_gcp_norm=use_gcp_norm
            )
            self.node_normalization = GCPLayerNorm(
                node_input_dims, use_gcp_norm=use_gcp_norm
            )
        else:
            self.edge_normalization = GCPLayerNorm(
                edge_hidden_dims, use_gcp_norm=use_gcp_norm
            )
            self.node_normalization = GCPLayerNorm(
                node_hidden_dims, use_gcp_norm=use_gcp_norm
            )

        self.edge_embedding = GCP(
            edge_input_dims,
            edge_hidden_dims,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

        self.node_embedding = GCP(
            node_input_dims,
            node_hidden_dims,
            nonlinearities=("none", "none"),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Tuple[
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
                Float[torch.Tensor, "batch_num_nodes m chi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
        ],
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
                Float[torch.Tensor, "batch_num_edges x xi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
        ],
    ]:
        if self.atom_embedding is not None:
            node_rep = ScalarVector(self.atom_embedding(batch.h), batch.chi)
        else:
            node_rep = ScalarVector(batch.h, batch.chi)

        edge_rep = ScalarVector(batch.e, batch.xi)

        edge_vectors = (
            batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)  # [n_edges, 1]
        edge_rep = ScalarVector(
            torch.cat((edge_rep.scalar, self.radial_embedding(edge_lengths)), dim=-1),
            edge_rep.vector,
        )

        edge_rep = (
            edge_rep.scalar if not self.edge_embedding.vector_input_dim else edge_rep
        )
        node_rep = (
            node_rep.scalar if not self.node_embedding.vector_input_dim else node_rep
        )

        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None),
        )
        node_rep = self.node_embedding(
            node_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=getattr(batch, "mask", None),
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        return node_rep, edge_rep


@typechecker
def get_GCP_with_custom_cfg(
    input_dims: Any, output_dims: Any, cfg: Any, **kwargs
):
    cfg_dict = _cfg_to_dict(cfg)
    if not isinstance(cfg_dict, dict):  # pragma: no cover - defensive guard
        raise TypeError("GCP configuration must be a mapping.")

    cfg_dict = copy(cfg_dict)
    cfg_dict.pop("scalar_nonlinearity", None)
    cfg_dict.pop("vector_nonlinearity", None)

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return GCP(input_dims, output_dims, **cfg_dict)


class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: Any,
        mp_cfg: Any,
        reduce_function: str = "sum",
        use_scalar_message_attention: bool = True,
    ):
        super().__init__()

        # hyperparameters
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.conv_cfg = mp_cfg
        self.self_message = self.conv_cfg.self_message
        self.reduce_function = reduce_function
        self.use_scalar_message_attention = use_scalar_message_attention
        self._cugraph_cache: Dict[Tuple[int, int, int, int], _CuGraphCSCCache] = {}

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        # config instantiations
        soft_cfg = copy(cfg)
        soft_cfg.bottleneck = cfg.default_bottleneck

        primary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=soft_cfg)
        secondary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=cfg)

        # PyTorch modules #
        module_list = [
            primary_cfg_GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims,
                nonlinearities=cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
        ]

        for _ in range(self.conv_cfg.num_message_layers - 2):
            module_list.append(
                secondary_cfg_GCP(
                    output_dims,
                    output_dims,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        if self.conv_cfg.num_message_layers > 1:
            module_list.append(
                primary_cfg_GCP(
                    output_dims,
                    output_dims,
                    nonlinearities=cfg.nonlinearities,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.message_fusion = nn.ModuleList(module_list)

        # learnable scalar message gating
        if use_scalar_message_attention:
            self.scalar_message_attention = nn.Sequential(
                nn.Linear(output_dims.scalar, 1), nn.Sigmoid()
            )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> ScalarVector:
        return self._fused_message_pass(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )

    def _fused_message_pass(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> ScalarVector:
        row, col = edge_index

        scalar_src = node_rep.scalar[row]
        scalar_dst = node_rep.scalar[col]

        vector_src = node_rep.vector[row]
        vector_dst = node_rep.vector[col]

        scalar_inputs = torch.cat((scalar_src, edge_rep.scalar, scalar_dst), dim=-1)

        if node_rep.vector.shape[1] == 0 and edge_rep.vector.shape[1] == 0:
            vector_inputs = edge_rep.vector
        else:
            vector_inputs = torch.cat((vector_src, edge_rep.vector, vector_dst), dim=1)

        message = ScalarVector(scalar_inputs, vector_inputs)

        message_residual = self.message_fusion[0](
            message, edge_index, frames, node_inputs=False, node_mask=node_mask
        )
        for module in self.message_fusion[1:]:
            new_message = module(
                message_residual,
                edge_index,
                frames,
                node_inputs=False,
                node_mask=node_mask,
            )
            message_residual = ScalarVector(
                message_residual.scalar + new_message.scalar,
                message_residual.vector + new_message.vector,
            )

        if self.use_scalar_message_attention:
            attention = self.scalar_message_attention(message_residual.scalar)
            message_residual = ScalarVector(
                message_residual.scalar * attention,
                message_residual.vector,
            )

        agg_scalar, agg_vector = self._sparse_reduce(
            row,
            col,
            message_residual.scalar,
            message_residual.vector,
            dim_size=node_rep.scalar.shape[0],
            node_mask=node_mask,
        )

        return ScalarVector(agg_scalar, agg_vector)

    def _cugraph_cache_key(
        self, row: torch.Tensor, col: torch.Tensor, num_nodes: int
    ) -> Tuple[int, int, int, int]:
        device = row.device
        device_index = device.index if device.type == "cuda" else -1
        return (
            int(row.data_ptr()),
            int(col.data_ptr()),
            num_nodes,
            device_index,
        )

    def _get_cugraph_cache(
        self, row: torch.Tensor, col: torch.Tensor, num_nodes: int
    ) -> _CuGraphCSCCache:
        if row.numel() == 0:
            return _build_cugraph_cache(row, col, num_nodes)

        key = self._cugraph_cache_key(row, col, num_nodes)
        cache = self._cugraph_cache.get(key)

        if cache is None or cache.num_edges != row.numel():
            cache = _build_cugraph_cache(row, col, num_nodes)
            # Retain only the most recent CSC graph to bound memory.
            self._cugraph_cache = {key: cache}

        return cache

    def _sparse_reduce(
        self,
        row: torch.Tensor,
        col: torch.Tensor,
        scalar_message: torch.Tensor,
        vector_message: torch.Tensor,
        *,
        dim_size: int,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_edges = row.numel()
        device = row.device

        if num_edges == 0:
            aggregated_scalar = scalar_message.new_zeros(dim_size, *scalar_message.shape[1:])
            if vector_message.numel():
                aggregated_vector = vector_message.new_zeros(dim_size, vector_message.size(1), vector_message.size(2))
            else:
                aggregated_vector = vector_message.new_zeros(dim_size, 0, 3)
            if node_mask is not None:
                mask = node_mask.to(aggregated_scalar.dtype).unsqueeze(-1)
                aggregated_scalar = aggregated_scalar * mask
                if aggregated_vector.numel():
                    aggregated_vector = aggregated_vector * mask.unsqueeze(-1)
            return aggregated_scalar, aggregated_vector

        use_cugraph = (
            _HAS_CUGRAPH_OPS
            and scalar_message.is_cuda
            and (vector_message.numel() == 0 or vector_message.is_cuda)
        )

        if use_cugraph:
            cache = self._get_cugraph_cache(row, col, dim_size)
            aggregated_scalar = _CuGraphSimpleE2N.apply(
                scalar_message, row, col, dim_size, cache
            )
            if vector_message.numel():
                vector_flat = vector_message.reshape(num_edges, -1)
                aggregated_vector_flat = _CuGraphSimpleE2N.apply(
                    vector_flat, row, col, dim_size, cache
                )
                aggregated_vector = aggregated_vector_flat.view(
                    dim_size, vector_message.size(1), vector_message.size(2)
                )
            else:
                aggregated_vector = vector_message.new_zeros(dim_size, 0, 3)
        else:
            scalar_dtype = scalar_message.dtype
            autocast_context = (
                (lambda: torch.amp.autocast(device_type="cuda", enabled=False))
                if scalar_message.is_cuda
                else contextlib.nullcontext
            )

            counts = torch.bincount(row, minlength=dim_size)
            indptr = torch.zeros(dim_size + 1, device=device, dtype=torch.long)
            indptr[1:] = counts.cumsum(0)

            perm = torch.argsort(row)
            scalar_message = scalar_message.index_select(0, perm)
            if vector_message.numel():
                vector_message = vector_message.index_select(0, perm)

            with autocast_context():
                scalar_fp32 = scalar_message.float()
                aggregated_scalar_fp32 = torch.segment_reduce(
                    scalar_fp32,
                    reduce="sum",
                    offsets=indptr,
                    axis=0,
                )

            aggregated_scalar = aggregated_scalar_fp32.to(scalar_dtype)

            if vector_message.numel():
                vector_flat = vector_message.reshape(num_edges, -1)
                with autocast_context():
                    vector_fp32 = vector_flat.float()
                    aggregated_vector_flat_fp32 = torch.segment_reduce(
                        vector_fp32,
                        reduce="sum",
                        offsets=indptr,
                        axis=0,
                    )
                aggregated_vector_flat = aggregated_vector_flat_fp32.to(vector_message.dtype)
                aggregated_vector = aggregated_vector_flat.view(
                    dim_size, vector_message.size(1), vector_message.size(2)
                )
            else:
                aggregated_vector = vector_message.new_zeros(dim_size, 0, 3)

        if node_mask is not None:
            mask = node_mask.to(aggregated_scalar.dtype).unsqueeze(-1)
            aggregated_scalar = aggregated_scalar * mask
            if aggregated_vector.numel():
                aggregated_vector = aggregated_vector * mask.unsqueeze(-1)

        return aggregated_scalar, aggregated_vector


class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: Any,
        layer_cfg: Any,
        dropout: float = 0.0,
        nonlinearities: Optional[Tuple[Any, Any]] = None,
    ):
        super().__init__()

        # hyperparameters #
        if nonlinearities is None:
            nonlinearities = cfg.nonlinearities
        self.pre_norm = layer_cfg.pre_norm
        self.predict_node_positions = getattr(cfg, "predict_node_positions", False)
        self.node_positions_weight = getattr(cfg, "node_positions_weight", 1.0)
        self.update_positions_with_vector_sum = getattr(
            cfg, "update_positions_with_vector_sum", False
        )
        reduce_function = "sum"

        # PyTorch modules #

        # geometry-complete message-passing neural network
        message_function = GCPMessagePassing

        self.interaction = message_function(
            node_dims,
            node_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg,
            reduce_function=reduce_function,
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention,
        )

        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_cfg)

        self.gcp_norm = nn.ModuleList(
            [GCPLayerNorm(node_dims, use_gcp_norm=layer_cfg.use_gcp_norm)]
        )
        self.gcp_dropout = nn.ModuleList(
            [GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout)]
        )

        # build out feedforward (FF) network modules
        hidden_dims = (
            (node_dims.scalar, node_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * node_dims.scalar, 2 * node_dims.vector)
        )
        ff_interaction_layers = [
            ff_GCP(
                (node_dims.scalar * 2, node_dims.vector * 2),
                hidden_dims,
                nonlinearities=("none", "none")
                if layer_cfg.num_feedforward_layers == 1
                else cfg.nonlinearities,
                feedforward_out=layer_cfg.num_feedforward_layers == 1,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
        ]

        interaction_layers = [
            ff_GCP(
                hidden_dims,
                hidden_dims,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_GCP(
                    hidden_dims,
                    node_dims,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # potentially build out node position update modules
        if self.predict_node_positions:
            # node position update GCP(s)
            position_output_dims = (
                node_dims
                if getattr(cfg, "update_positions_with_vector_sum", False)
                else (node_dims.scalar, 1)
            )
            self.node_position_update_gcp = ff_GCP(
                node_dims,
                position_output_dims,
                nonlinearities=cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )

    @jaxtyped(typechecker=typechecker)
    def derive_x_update(
        self,
        node_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        f_ij: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Float[torch.Tensor, "batch_num_nodes 3"]:
        # use vector-valued features to derive node position updates
        node_rep_update = self.node_position_update_gcp(
            node_rep, edge_index, f_ij, node_inputs=True, node_mask=node_mask
        )
        if self.update_positions_with_vector_sum:
            x_vector_update = node_rep_update.vector.sum(1)
        else:
            x_vector_update = node_rep_update.vector.squeeze(1)

        # (up/down)weight position updates
        x_update = x_vector_update * self.node_positions_weight

        return x_update

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        node_rep: Tuple[
            Float[torch.Tensor, "batch_num_nodes node_hidden_dim"],
            Float[torch.Tensor, "batch_num_nodes m 3"],
        ],
        edge_rep: Tuple[
            Float[torch.Tensor, "batch_num_edges edge_hidden_dim"],
            Float[torch.Tensor, "batch_num_edges x 3"],
        ],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
        node_pos: Optional[Float[torch.Tensor, "batch_num_nodes 3"]] = None,
    ) -> Tuple[
        Tuple[
            Float[torch.Tensor, "batch_num_nodes hidden_dim"],
            Float[torch.Tensor, "batch_num_nodes n 3"],
        ],
        Optional[Float[torch.Tensor, "batch_num_nodes 3"]],
    ]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # forward propagate with interaction module
        hidden_residual = self.interaction(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )

        # aggregate input and hidden features
        hidden_residual = ScalarVector(
            torch.cat((hidden_residual.scalar, node_rep.scalar), dim=-1),
            torch.cat((hidden_residual.vector, node_rep.vector), dim=1),
        )

        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                edge_index,
                frames,
                node_inputs=True,
                node_mask=node_mask,
            )

        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout[0](hidden_residual)

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask.float())

        # bypass updating node positions
        if not self.predict_node_positions:
            return node_rep, node_pos

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames, node_mask=node_mask
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        return node_rep, node_pos
