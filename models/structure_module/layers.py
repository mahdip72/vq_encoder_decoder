import torch.nn as nn


def activation_func(activation, inplace=False):
    """
    Activation functions
    """
    if activation is None:
        return None
    return nn.ModuleDict(
        [
            ["relu", nn.ReLU(inplace=inplace)],
            ["elu", nn.ELU(inplace=inplace)],
            ["leaky_relu", nn.LeakyReLU(inplace=inplace)],
            ["selu", nn.SELU(inplace=inplace)],
            ["none", nn.Identity()],
        ]
    )[activation]


def normalization_func(input_size, normalization, n_dim):
    """
    Normalization functions
    """
    assert input_size in ["1D", "2D"], "input_size: 1D or 2D."
    if input_size == "1D":
        return nn.ModuleDict(
            [
                ["batch", nn.BatchNorm1d(n_dim)],
                ["instance", nn.InstanceNorm1d(n_dim)],
                ["layer", nn.LayerNorm(n_dim)],
                ["none", nn.Identity()],
            ]
        )[normalization]

    elif input_size == "2D":
        return nn.ModuleDict(
            [
                ["batch", nn.BatchNorm2d(n_dim)],
                ["instance", nn.InstanceNorm2d(n_dim)],
                ["none", nn.Identity()],
            ]
        )[normalization]


class FeedForward(nn.Module):
    """
    Feed Forward Layer.
    """

    def __init__(
        self,
        n_input,
        n_output,
        n_hidden=None,
        activation="relu",
        bias=False,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        w1_out = n_output if n_hidden is None else n_hidden
        self.w_1 = nn.Linear(n_input, w1_out, bias=bias)
        self.activation = activation_func(activation)
        if n_hidden is not None:
            self.w_2 = nn.Linear(n_hidden, n_output, bias=bias)

    def forward(self, x, *args, **kwargs):
        x = self.w_1(x)
        x = self.activation(x)
        if self.n_hidden is not None:
            x = self.w_2(x)
        return x


class Residual(nn.Module):
    """
    Residual wrapper.
    """

    def __init__(self, fn, n_input):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(n_input)

    def forward(self, x, *args, **kwargs):
        residual = x

        x = self.norm(x)

        outputs = self.fn(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x