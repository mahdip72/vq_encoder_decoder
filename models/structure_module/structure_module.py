import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import numpy as np
from models.structure_module.pairwise import Pairwise

from models.structure_module.side_chain import SideChain1D


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        seq_dim_in,
        pair_dim_in,
        scalar_qk_dim,
        scalar_v_dim,
        point_qk_num,
        point_v_num,
        n_head,
        dist_epsilon=1e-8,
    ):

        super().__init__()

        self.n_head = n_head
        self.scalar_qk_dim = scalar_qk_dim
        self.scalar_v_dim = scalar_v_dim

        self.point_qk_num = point_qk_num
        self.point_v_num = point_v_num

        self.seq_dim_in = seq_dim_in
        self.pair_dim_in = pair_dim_in

        self.dist_epsilon = dist_epsilon

        # here we follow the source code of AF2, instead of its paper
        self.to_sq = nn.Linear(seq_dim_in, n_head * scalar_qk_dim)
        self.to_sk = nn.Linear(seq_dim_in, n_head * scalar_qk_dim)
        self.to_sv = nn.Linear(seq_dim_in, n_head * scalar_v_dim)

        self.to_sb = nn.Linear(pair_dim_in, n_head)

        self.to_pq = nn.Linear(seq_dim_in, n_head * point_qk_num * 3)
        self.to_pk = nn.Linear(seq_dim_in, n_head * point_qk_num * 3)
        self.to_pv = nn.Linear(seq_dim_in, n_head * point_v_num * 3)

        self.attn_linear = nn.Linear(3, 1)

        scalar_variance = max(scalar_qk_dim, 1) * 1.0
        point_variance = max(point_qk_num, 1) * 9.0 / 2

        num_logit_terms = 3

        self.scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
        self.point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance))
        self.attention_2d_weights = np.sqrt(1.0 / (num_logit_terms))

        self.trainable_point_weight = nn.Parameter(torch.ones(n_head))
        nn.init.constant_(self.trainable_point_weight, np.log(np.exp(1.0) - 1.0))

        self.output_layer = nn.Linear(
            n_head * (scalar_v_dim + point_v_num * 4 + pair_dim_in), seq_dim_in
        )

    def forward(self, s, z, rot, tran, res_mask):
        """
        Input:
            s : [B, N, C_s]
            z : [B, N, N, C_z]
            rot: [B, N, 3, 3]
            tran: [B, N, 3]
            res_mask: [B, N]

        B: batch_size
        N: seq_len
        C_s: seq_dim_in
        C_z: pair_dim_in

        """

        # scalar query, key, value
        sq, sk, sv = self.to_sq(s), self.to_sk(s), self.to_sv(s)
        sq, sk, sv = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.n_head), (sq, sk, sv)
        )

        # pair scalar bias
        sb = self.to_sb(z)  # [ b, i, j, h]

        # point query, key, value
        pq, pk, pv = self.to_pq(s), self.to_pk(s), self.to_pv(s)
        pq, pk, pv = map(
            lambda t: rearrange(t, "b n (h p d) -> b n h p d", h=self.n_head, d=3),
            (pq, pk, pv),
        )

        # apply rotation on query, key, value
        pq_global, pk_global, pv_global = map(
            lambda t: einsum(" b n x y, b n h p y -> b n h p x", rot, t), (pq, pk, pv)
        )  # [b n h p 3]
        pq_global, pk_global, pv_global = map(
            lambda t: t + repeat(tran, "b n c -> b n () () c").expand_as(t),
            (pq_global, pk_global, pv_global),
        )
        pv_global = rearrange(pv_global, "b n h p c -> (b h) n p c")

        # point attention logits
        pq_global, pk_global = map(
            lambda t: rearrange(t, "b n h p c -> (b h) n p c"), (pq_global, pk_global)
        )
        diff = rearrange(pq_global, "bh i p c -> bh i () p c") - rearrange(
            pk_global, "bh j p c -> bh () j p c"
        )  # [bh i j p 3]
        square_dist = diff.square().sum(dim=-1)  # [bh i j p]
        square_dist_sum = square_dist.sum(dim=-1)  # [bh i j]
        square_dist_sum = rearrange(
            square_dist_sum, " (b h) i j -> b i j h", h=self.n_head
        )

        attn_qk_point = (
            -0.5
            * F.softplus(self.trainable_point_weight)
            * square_dist_sum
            * self.point_weights
        )  # [ b i j h]

        # scalar attention logits
        attn_qk_scalar = einsum(
            "b h i d, b h j d -> b i j h", sq * self.scalar_weights, sk
        )  # [ b i j h]

        # pair attention logits
        attn_pair = sb * self.attention_2d_weights  # [ b i j h]
        # attn_logits = attn_qk_scalar + attn_pair + attn_qk_point # [b, i, j, h]

        attn_cat = torch.stack(
            [attn_qk_scalar, attn_qk_point, attn_pair], dim=-1
        )  # [b i j h 3]
        attn_logits = self.attn_linear(attn_cat).squeeze(-1)  # [b i j h]

        # mask attention logits
        mask_2d = res_mask[:, :, None] * res_mask[:, None, :]
        attn_logits.masked_fill_(
            mask_2d[:, :, :, None] == 0, torch.finfo(attn_logits.dtype).min
        )

        # calc attention
        attn_logits = rearrange(attn_logits, "b i j h -> (b h) i j")
        attn = F.softmax(attn_logits, dim=-1)  # [ bh, i, j]

        # scalar with attention
        sv = rearrange(sv, "b h j d -> (b h) j d")
        result_scalar = einsum(
            " b i j, b j d -> b i d", attn, sv
        )  # bh i j, bh j d -> bh i d

        # point with attention
        result_point_global = einsum(
            "b i j, b j p c -> b i p c", attn, pv_global
        )  # bh i j ,bh j p 3 -> bh i p 3
        rot_ext = repeat(rot, "b i x y -> (b h) i x y", h=self.n_head)
        tran_ext = repeat(tran, "b i x -> (b h) i x", h=self.n_head)
        result_point = einsum(
            "b i x y, b i p y -> b i p x",
            rot_ext.transpose(-1, -2),
            result_point_global - repeat(tran_ext, "b i c -> b i () c"),
        )
        # 'bh i x y, bh i p y -> bh i p 3'  - bh i p 3

        # point with attention norm
        result_point_norm = (
            result_point.square().sum(dim=-1) + self.dist_epsilon
        ).sqrt()  # [bh i p]
        result_point = rearrange(result_point, "bh i p x -> bh i (p x)")

        # pair with attention'
        attn = rearrange(attn, "(b h) i j -> b h i j", h=self.n_head)
        result_attn_pair = einsum("b h i j, b i j c -> b h i c", attn, z)
        result_attn_pair = result_attn_pair.reshape((-1,) + result_attn_pair.shape[2:])

        result = torch.cat(
            [result_scalar, result_point, result_point_norm, result_attn_pair], dim=-1
        )
        # [bh i d] + [bh i px] + [bh i p] + [bh i c] -> [bh i (d + px + p + c)]
        result = rearrange(result, "(b h) i d -> b i (h d)", h=self.n_head)
        result = self.output_layer(result)  # [B N C_s]

        return result


def BackBoneUpdate(s, linear):
    """

    s: [B, N, C_s]
    """

    bcd, tran = linear(s).split([3, 3], dim=-1)

    abcd = torch.cat([torch.ones_like(bcd[..., [0]]), bcd], dim=-1)
    a, b, c, d = F.normalize(abcd, dim=-1).unbind(-1)

    rot = torch.stack(
        [
            a.square() + b.square() - c.square() - d.square(),
            2 * b * c - 2 * a * d,
            2 * b * d + 2 * a * c,
            2 * b * c + 2 * a * d,
            a.square() - b.square() + c.square() - d.square(),
            2 * c * d - 2 * a * b,
            2 * b * d - 2 * a * c,
            2 * c * d + 2 * a * b,
            a.square() - b.square() - c.square() + d.square(),
        ],
        dim=-1,
    )

    rot = rot.reshape(rot.shape[:-1] + (3, 3))
    return rot, tran


def merge_rotran(rot1, tran1, rot2, tran2):
    """
    rot: [B, N, 3, 3]
    tran: [B, N, 3]

    nrot:  rot1 @ rot2
    ntran: rot1 @ tran2 + tran1
    """

    rot = einsum("b n x y, b n y z -> b n x z", rot1, rot2)
    tran = einsum("b n x y, b n y -> b n x", rot1, tran2) + tran1

    return rot, tran


class PredictedLDDT(nn.Module):
    def __init__(self, node_dim_in, node_dim_hidden=128, num_bins=50):
        super().__init__()

        self.norm = nn.LayerNorm(node_dim_in)

        self.lin1 = nn.Linear(node_dim_in, node_dim_hidden)
        self.lin2 = nn.Linear(node_dim_hidden, node_dim_hidden)
        self.lin3 = nn.Linear(node_dim_hidden, num_bins)
        self.relu = nn.ReLU()

    def forward(self, node_feat):
        x = self.norm(node_feat)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x_out = self.lin3(x)

        return x_out


class StrucModule(nn.Module):
    def __init__(
        self,
        node_dim_in=256,
        edge_dim_in=128,
        node_dim_hidden=384,
        edge_dim_hidden=128,
        scalar_qk_dim=16,
        scalar_v_dim=16,
        point_qk_num=4,
        point_v_num=8,
        n_head=12,
        n_layer=8,
        dropout=0.1,
        position_scale=10.0,
        *args,
        **kwargs
    ):
        super().__init__()

        self.n_layer = n_layer
        self.position_scale = position_scale

        # project layer
        self.seq_proj = (
            nn.Linear(node_dim_in, node_dim_hidden)
            if node_dim_in != node_dim_hidden
            else None
        )
        self.pair_proj = (
            nn.Linear(edge_dim_in, edge_dim_hidden)
            if edge_dim_in != edge_dim_hidden
            else None
        )

        # init seq
        self.init_seq_norm = nn.LayerNorm(node_dim_hidden)
        self.init_seq_linear = nn.Linear(node_dim_hidden, node_dim_hidden)

        # init pair
        self.init_pair_norm = nn.LayerNorm(edge_dim_hidden)

        # IPA
        self.IPA = InvariantPointAttention(
            node_dim_hidden,
            edge_dim_hidden,
            scalar_qk_dim,
            scalar_v_dim,
            point_qk_num,
            point_v_num,
            n_head,
        )
        self.IPA_seq_norm = nn.LayerNorm(node_dim_hidden)
        self.IPA_drop = nn.Dropout(dropout)

        # transition
        self.transition = nn.Sequential(
            nn.Linear(node_dim_hidden, node_dim_hidden),
            nn.ReLU(),
            nn.Linear(node_dim_hidden, node_dim_hidden),
            nn.ReLU(),
            nn.Linear(node_dim_hidden, node_dim_hidden),
        )
        self.trans_seq_norm = nn.LayerNorm(node_dim_hidden)
        self.trans_drop = nn.Dropout(dropout)

        # Update backbone
        self.bbupdate_layer = nn.Linear(node_dim_hidden, 6)

        # # side chain
        # self.sidechain_module = SideChain1D(node_dim_hidden, node_dim_hidden)
        #
        # # pLDDT
        # self.pLDDT_module = PredictedLDDT(node_dim_hidden)

    def forward(self, seq_feat, pair_feat, res_mask):
        """
        seq_feat: [B, L, C_seq]
        pair_feat: [B, L, L, C_pair]
        res_mask: [B, L]
        """
        B, L, _ = seq_feat.shape

        if self.seq_proj is not None:
            seq_feat = self.seq_proj(seq_feat)
        if self.pair_proj is not None:
            pair_feat = self.pair_proj(pair_feat)

        # init seq
        seq_feat_init = self.init_seq_norm(seq_feat)
        seq_feat = self.init_seq_linear(seq_feat_init)

        # init pair
        pair_feat = self.init_pair_norm(pair_feat)

        # init rotran
        rot = repeat(torch.eye(3), "x y -> b l x y", b=B, l=L).type_as(seq_feat)
        tran = repeat(torch.zeros(3), "x -> b l x", b=B, l=L).type_as(seq_feat)

        # iteration
        ca_coords, rots = [], []
        # unnorm_angles, norm_angles = [], []

        for i in range(self.n_layer):
            seq_feat = seq_feat + self.IPA(seq_feat, pair_feat, rot, tran, res_mask)
            seq_feat = self.IPA_seq_norm(self.IPA_drop(seq_feat))
            seq_feat = seq_feat + self.transition(seq_feat)
            seq_feat = self.trans_seq_norm(self.trans_drop(seq_feat))

            nrot, ntran = BackBoneUpdate(seq_feat, self.bbupdate_layer)
            rot, tran = merge_rotran(rot, tran, nrot, ntran)

            ca_coords.append(tran)
            rots.append(rot)

            # # predict side_chain
            # norm_angle, unnorm_angle = self.sidechain_module(seq_feat, seq_feat_init)
            # unnorm_angles.append(unnorm_angle)
            # norm_angles.append(norm_angle)

            rot = rot.detach()

        # main chain
        outputs = (
            ca_coords[-1] * self.position_scale,
            rots[-1],
            torch.stack(ca_coords) * self.position_scale,
            torch.stack(rots),
        )

        # # side chain
        # outputs = outputs + (
        #     unnorm_angles[-1],
        #     norm_angles[-1],
        #     torch.stack(unnorm_angles),
        #     torch.stack(norm_angles),
        # )

        # # pLDDT
        # pLDDT = self.pLDDT_module(seq_feat)
        # outputs = outputs + (pLDDT,)

        return outputs


if __name__ == '__main__':

    batch_size = 1
    N = 100 # Protein length
    input_dim = 256  # Input embedding dimension
    embedding_dim = 128  # Output embedding dimension
    hidden_dim = 64  # Hidden dimension for MLP

    # Create a batch of embeddings with shape (N, input_dim)
    seq_feat = torch.randn(batch_size, N, input_dim)

    # Instantiate the model
    pairwise_model = Pairwise(input_dim=input_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    # Forward pass to get k
    pair_feat = pairwise_model(seq_feat)

    struc_model = StrucModule(node_dim_in=input_dim, edge_dim_in=embedding_dim)
    # seq_feat = torch.randn(1, 100, 256)
    # pair_feat = torch.randn(1, 100, 100, 128)

    print("seq_feat", seq_feat.shape, "pair_feat", pair_feat.shape)
    # exit()

    res_mask = torch.ones(batch_size, N)
    outputs = struc_model(seq_feat, pair_feat, res_mask)
    for out in outputs:
        print(out.shape)