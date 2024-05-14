import torch, functools
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vector_quantize_pytorch import VectorQuantize, LFQ
# import gvp.models
from torch_geometric.nn import radius, global_mean_pool, global_max_pool
from data import *
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_scatter import scatter_add


def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]


def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)

    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
        torch.randn(n, dims[1], 3, device=device)


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                        activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index,
                                 s=x_s, v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):

        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean",
                            activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )

            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class TransformersVQAutoEncoder(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_encoder_layers=6, dim_feedforward=64, **kwargs):
        super().__init__()

        self.d_model = kwargs['dim']
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward

        # Initial Convolution Block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Transformer Encoder Block
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, activation='gelu', batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.positional_encoding = nn.Parameter(torch.randn(1, 196, self.d_model))

        # Vector Quantization Layer as before
        self.vq_layer = VectorQuantize(
            dim=self.d_model,
            codebook_size=kwargs['codebook_size'],
            decay=kwargs['decay'],
            commitment_weight=kwargs['commitment_weight'],
            accept_image_fmap=True
        )

        # Decoder Convolution Block
        self.decoder_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Conv2d(self.d_model, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),

        ])

    def forward(self, x, return_vq_only=False):
        x = self.initial_conv(x)
        n, c, h, w = x.shape
        x = x.view(n, h * w, c)  # Adjust to (batch_size, seq_length, feature_size)
        x += self.positional_encoding[:, :h * w, :]  # Ensure positional encoding is added correctly
        x = self.transformer_encoder(x)
        x = x.view(n, c, h, w)  # Reshape back to feature map

        x, indices, commit_loss = self.vq_layer(x)

        if return_vq_only:
            return x, indices, commit_loss

        for layer in self.decoder_layers:
            x = layer(x)

        return x.clamp(-1, 1), indices, commit_loss


class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.d_model = kwargs['dim']

        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            # nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.vq_layer = VectorQuantize(
            dim=self.d_model,
            codebook_size=kwargs['codebook_size'],
            decay=kwargs['decay'],
            commitment_weight=kwargs['commitment_weight'],
            accept_image_fmap=True
        )
        # from math import log2
        # self.vq_layer = LFQ(
        #     dim=self.d_model,
        #     codebook_size=kwargs['codebook_size'],
        #     entropy_loss_weight=0.02,
        #     diversity_gamma=1.0
        # )

        self.decoder_layers = nn.ModuleList([
            # nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv2d(self.d_model, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x, return_vq_only=False):
        for layer in self.encoder_layers:
            x = layer(x)
            # print(x.shape)

        x, indices, commit_loss = self.vq_layer(x)

        if return_vq_only:
            return x, indices, commit_loss

        for layer in self.decoder_layers:
            x = layer(x)
            # print(x.shape)
        # make sure the output is in the range of 0-1
        x = torch.clamp(x, 0, 1)
        return x, indices, commit_loss


class structure_encoder(nn.Module):
    '''
    GVP-GNN for protein sequence representation, inspired by gearnet/gvp.py
    modified based on GVP-GNN for Model Quality Assessment as described in manuscript.

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]

    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.

    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers, out_dims
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''

    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim,
                 seq_in=False, seq_embed_mode="embedding", seq_embed_dim=20,
                 num_layers=3, drop_rate=0.1,
                 vector_gate=True,
                 ):

        super(structure_encoder, self).__init__()
        self.seq_embed_mode = seq_embed_mode
        if seq_in and seq_embed_mode == "embedding":
            self.W_s = nn.Embedding(20, seq_embed_dim)
            node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])
        elif seq_in:
            node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])

        # if seq_in and seq_embed_mode=="PhysicsPCA":
        #    node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])
        # if seq_in and seq_embed_mode=="ESM2":
        #    node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        # GVPConvLayer is a nn.Module that forms messages using a GVPConv and updates the node embeddings as described in the paper
        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2 * ns, 1)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        # print(seq)
        if seq is not None:
            if len(seq.shape) == 1:
                seq = self.W_s(seq)
                h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
            else:  # seq representation from ESM2, PhysicsPCA or Atchleyfactor
                h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)  # in gearnet added h_edge = self.rbf((pos_out - pos_in).norm(dim=-1)), vec_edge why?
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        # if batch is None: out = out.mean(dim=0, keepdims=True)
        # else: out = scatter_mean(out, batch, dim=0)

        return out


class GVPEncoder(nn.Module):  # embedding table can be tuned
    def __init__(self, residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096,
                 protein_num_projector=2,
                 seqlen=512):
        super(GVPEncoder, self).__init__()
        node_in_dim = [6, 3]  # default

        node_in_dim = tuple(node_in_dim)
        edge_in_dim = (
            16 + 16,
            1)  # num_rbf+num_positional_embeddings

        node_h_dim = (100, 16)  # default
        # node_h_dim = (100, 32)  # seems best?
        edge_h_dim = (32, 1)  # default
        gvp_num_layers = 3

        self.use_seq = False
        if self.use_seq:
            self.seq_embed_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
            self.backbone = structure_encoder(node_in_dim, node_h_dim,
                                              edge_in_dim, edge_h_dim, seq_in=True,
                                              seq_embed_mode=self.seq_embed_mode,
                                              seq_embed_dim=configs.model.struct_encoder.use_seq.seq_embed_dim,
                                              num_layers=gvp_num_layers)
        else:
            self.backbone = structure_encoder(node_in_dim, node_h_dim,
                                              edge_in_dim, edge_h_dim, seq_in=False,
                                              num_layers=gvp_num_layers)

        # pretrained_state_dict = init_model.state_dict()
        # self.backbone.load_state_dict(pretrained_state_dict,strict=False)

    def forward(self, graph,
                esm2_representation=None):  # this batch is torch_geometric batch.batch to indicate the batch
        """
        graph: torch_geometric batchdasta
        """
        nodes = (graph['graph'].node_s, graph['graph'].node_v)
        edges = (graph['graph'].edge_s, graph['graph'].edge_v)
        if self.use_seq and self.seq_embed_mode != "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph['graph'].edge_index, edges,
                                                      seq=graph.seq)
        elif self.use_seq and self.seq_embed_mode == "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph['graph'].edge_index, edges,
                                                      seq=esm2_representation
                                                      )
        else:
            residue_feature_embedding = self.backbone(nodes, graph['graph'].edge_index, edges,
                                                      seq=None)

        # graph_feature=scatter_mean(residue_feature, batch, dim=0)
        graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph['graph'].batch)
        return graph_feature_embedding, residue_feature_embedding


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model, logging, description=""):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"{description} trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


def prepare_models():
    residue_inner_dim = 4096,
    residue_out_dim = 256,
    protein_out_dim = 256,
    residue_num_projector = 2,
    protein_inner_dim = 4096,
    protein_num_projector = 2,
    seqlen = 512
    gvp_model = GVPEncoder(residue_inner_dim=4096,
                           residue_out_dim=256,
                           protein_out_dim=256,
                           residue_num_projector=2,
                           protein_inner_dim=4096,
                           protein_num_projector=2,
                           seqlen=512)

    # model = SimpleVQAutoEncoder(
    #     dim=configs.model.vector_quantization.dim,
    #     codebook_size=configs.model.vector_quantization.codebook_size,
    #     decay=configs.model.vector_quantization.decay,
    #     commitment_weight=configs.model.vector_quantization.commitment_weight
    # )

    # if accelerator.is_main_process:
    #     print_trainable_parameters(model, logging, 'VQ-VAE')

    return gvp_model


if __name__ == '__main__':
    import yaml
    from utils import load_configs

    config_path = "./config.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main_configs = load_configs(config_file)

    net = SimpleVQAutoEncoder(
        dim=main_configs.model.vector_quantization.dim,
        codebook_size=main_configs.model.vector_quantization.codebook_size,
        decay=main_configs.model.vector_quantization.decay,
        commitment_weight=main_configs.model.vector_quantization.commitment_weight
    )

    dataset_path = './data/h5'  # test for piece of data
    dataset = ProteinGraphDataset(dataset_path)

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    for batch in test_dataloader:
        model = prepare_models()
        output = model(batch)
        print(output)

    # create a random input tensor and pass it through the network
    # x = torch.randn(1, 3, 32, 32)
    # output, x, y = net(x, return_vq_only=False)
    # print(output.shape)
    # print(x.shape)
    # print(y.shape)
