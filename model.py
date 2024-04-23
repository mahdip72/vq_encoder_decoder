import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vector_quantize_pytorch import VectorQuantize


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
        x += self.positional_encoding[:, :h*w, :]  # Ensure positional encoding is added correctly
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
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.vq_layer = VectorQuantize(
            dim=kwargs['dim'],
            codebook_size=kwargs['codebook_size'],  # codebook size
            decay=kwargs['decay'],
            commitment_weight=kwargs['commitment_weight'],
            accept_image_fmap=True
        )
        self.decoder_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x, return_vq_only=False):
        for layer in self.encoder_layers:
            x = layer(x)

        x, indices, commit_loss = self.vq_layer(x)

        if return_vq_only:
            return x, indices, commit_loss

        for layer in self.decoder_layers:
            x = layer(x)

        return x.clamp(-1, 1), indices, commit_loss


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


def prepare_models(configs, logging, accelerator):

    model = SimpleVQAutoEncoder(
        dim=configs.model.vector_quantization.dim,
        codebook_size=configs.model.vector_quantization.codebook_size,
        decay=configs.model.vector_quantization.decay,
        commitment_weight=configs.model.vector_quantization.commitment_weight
    )

    if accelerator.is_main_process:
        print_trainable_parameters(model, logging, 'VQ-VAE')

    return model


if __name__ == '__main__':
    import yaml
    from utils import load_configs

    config_path = "./config.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main_configs = load_configs(config_file)

    net = TransformersVQAutoEncoder(
        dim=main_configs.model.vector_quantization.dim,
        codebook_size=main_configs.model.vector_quantization.codebook_size,
        decay=main_configs.model.vector_quantization.decay,
        commitment_weight=main_configs.model.vector_quantization.commitment_weight
    )

    # create a random input tensor and pass it through the network
    x = torch.randn(1, 1, 28, 28)
    output, x, y = net(x, return_vq_only=False)
    print(output.shape)
    print(x.shape)
    print(y.shape)
