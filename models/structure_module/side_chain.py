import torch.nn as nn
import torch.nn.functional as F

from models.structure_module.resnet_1d import ResNet1DBlock


class SideChain(nn.Module):
    def __init__(self, node_dim_in=128, sc_dim_hidden=128, num_residual_block=2):
        super().__init__()

        self.relu = nn.ReLU()

        self.lin_s_cur = nn.Linear(node_dim_in, sc_dim_hidden)
        self.lin_s_init = nn.Linear(node_dim_in, sc_dim_hidden)

        res_blocks = [
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(sc_dim_hidden, sc_dim_hidden),
                nn.ReLU(),
                nn.Linear(sc_dim_hidden, sc_dim_hidden),
            )
            for _ in range(num_residual_block)
        ]

        self.res_blocks = nn.ModuleList(res_blocks)

        self.out_layer = nn.Sequential(nn.ReLU(), nn.Linear(sc_dim_hidden, 14))

    def forward(self, s_cur, s_init, rot, tran):

        """
        s_cur: [B, N, C_s]
        s_init:[B, N, C_s]
        rot: [B, N, 3, 3]
        tran: [B, N, 1, 3]
        """
        B, N, C_s = s_cur.shape
        a = self.lin_s_cur(self.relu(s_cur)) + self.lin_s_init(
            self.relu(s_init)
        )  # [B, N, dim_hidden]

        for res_block in self.res_blocks:
            res = res_block(a)
            a = a + res

        a_out = self.out_layer(a)
        a_unnorm = a_out.reshape((B, N, 7, 2))
        a_norm = F.normalize(a_unnorm, dim=-1)

        return a_norm, a_unnorm


class SideChain1D(nn.Module):
    """
    Use 1D Conv layer for side chain prediction
    """

    def __init__(
        self,
        node_dim_in=128,  # dim of input feature
        sc_dim_hidden=128,  # dim of hidden layer
        num_residual_block=4,  # number of residuel block
        kernel_size=3,
        dilation=1,
        dropout=0.1,
        normalization="instance",
        activation="relu",
        bias=False,
    ):

        super().__init__()

        self.relu = nn.ReLU()

        self.lin_s_cur = nn.Linear(node_dim_in, sc_dim_hidden)
        self.lin_s_init = nn.Linear(node_dim_in, sc_dim_hidden)

        self.blocks = nn.ModuleList()
        for i in range(num_residual_block):
            layer = ResNet1DBlock(
                sc_dim_hidden,
                sc_dim_hidden,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                activation=activation,
                normalization=normalization,
                bias=bias,
            )

            self.blocks.append(layer)

        self.out_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(sc_dim_hidden, 14),
        )

    def forward(self, s_cur, s_init):
        """
        s_cur: [B, N, C_s]
        s_init:[B, N, C_s]
        """
        B, N, _ = s_cur.shape
        x = self.lin_s_cur(self.relu(s_cur)) + self.lin_s_init(
            self.relu(s_init)
        )  # [B, N, dim_hidden]
        x = x.transpose(-1, -2)  # [B, dim_hidden, N]
        for block in self.blocks:
            x = block(x)

        x = x.transpose(-1, -2)  # [B, dim_hidden, N] -> [B, N, dim_hidden]
        a_out = self.out_layer(x)
        a_unnorm = a_out.reshape((B, N, 7, 2))
        a_norm = F.normalize(a_unnorm, dim=-1)

        return a_norm, a_unnorm