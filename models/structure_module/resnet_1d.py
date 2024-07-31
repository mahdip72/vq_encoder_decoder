import torch
import torch.nn as nn

from models.structure_module.layers import activation_func, normalization_func


class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding = (self.dilation[0] * (self.kernel_size[0] - 1) // 2,)


class ResNet1DBlock(nn.Module):
    """
    ResNet Residual Block
    """

    def __init__(
        self,
        n_input,
        n_output,
        kernel_size=5,
        dilation=2,
        dropout=0.0,
        activation="elu",
        normalization="instance",
        bias=False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.skip_connection = (
            nn.Sequential(
                Conv1dAuto(n_input, n_output, kernel_size=1, bias=bias),
                normalization_func("1D", normalization, n_output),
            )
            if n_input != n_output
            else None
        )

        self.block = nn.Sequential(
            Conv1dAuto(
                n_input, n_output, kernel_size=kernel_size, dilation=dilation, bias=bias
            ),
            normalization_func("1D", normalization, n_output),
            activation_func(activation),
            nn.Dropout(p=dropout),
            Conv1dAuto(
                n_output,
                n_output,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=bias,
            ),
            normalization_func("1D", normalization, n_output),
        )

        self.activate = activation_func(activation)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.block(x)
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
        x += residual
        x = self.activate(x)
        return x
