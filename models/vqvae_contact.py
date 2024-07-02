import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch import LFQ
from tqdm import tqdm

import yaml
from utils.utils import get_dummy_logger
from torch.utils.data import DataLoader
from accelerate import Accelerator
from box import Box

from data.data_contactmap import prepare_dataloaders


def add_encoder_layer(module_list, first_layer, last_layer, configs):
    """
    Add a layer to the encoder with 2D convolution, 2D batch normalization, ReLU activation,
    and 2D max pooling. If the layer is the first layer, the number of input channels must be 3.
    If the layer is the last layer, the number of output channels must match the dimension of the
    vector quantization layer.

    :param module_list: (list) list to which to add nn.Modules
    :param first_layer: (bool) True if the layer is the first layer, False otherwise
    :param last_layer: (bool) True if the layer is the last layer, False otherwise
    :param configs: (Box) configurations for the model
    :return: None
    """

    in_channels = 0
    out_channels = 0

    # in_channels = 3 for first layer, encoder.dim for all other layers
    if first_layer:
        in_channels = 3
    else:
        in_channels = configs.model.encoder.dim

    # out_channels = vq.dim for last layer, encoder.dim for all other layers
    if last_layer:
        out_channels = configs.model.vector_quantization.dim
    else:
        out_channels = configs.model.encoder.dim

    module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
    module_list.append(nn.BatchNorm2d(out_channels))
    module_list.append(nn.ReLU())

def add_decoder_layer(module_list, first_layer, last_layer, configs):
    """
    Add a layer to the decoder with upsample, 2D convolution, 2D batch normalization,
    and ReLU activation. If the layer is the first layer, the number of input channels must match
    the dimension of the vector quantization layer. If the layer is the last layer, the number of
    output channels must be 3, and the activation function must be tanh to get outputs in the range
    [-1, 1].

    :param module_list: (list) list to which to add nn.Modules
    :param first_layer: (bool) True if the layer is the first layer, False otherwise
    :param last_layer: (bool) True if the layer is the last layer, False otherwise
    :param configs: (Box) configurations for the model
    :return: None
    """

    in_channels = 0
    out_channels = 0

    # in_channels = vq.dim for first layer, decoder.dim for all other layers
    if first_layer:
        in_channels = configs.model.vector_quantization.dim
    else:
        in_channels = configs.model.decoder.dim

    # out_channels = 3 for last layer, decoder.dim for all other layers
    if last_layer:
        out_channels = 3
    else:
        out_channels = configs.model.decoder.dim

    module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
    module_list.append(nn.BatchNorm2d(out_channels))

    # Add activation function
    if last_layer:
        module_list.append(nn.Tanh())
    else:
        module_list.append(nn.ReLU())



def get_layers(encoder, num_layers, configs):
    """
    Get a list of layers for either the encoder or the decoder of a VQVAE model.
    :param encoder: (bool) True for encoder, False for decoder
    :param num_layers: (int) number of layers to include
    :param configs: (Box) configurations for the model
    :return: (list[nn.Module]) list of layers
    """

    layers = []

    # Add layers to ModuleList
    for i in range(num_layers):
        first_layer = False # Whether current layer is the first layer
        last_layer = False # Whether current layer is the last layer

        if i == 0:
            first_layer = True
        if i >= num_layers - 1:
            last_layer = True

        if encoder:
            add_encoder_layer(layers, first_layer, last_layer, configs)
        else:
            add_decoder_layer(layers, first_layer, last_layer, configs)

    return layers


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + residual

        return x


class VQVAEResNet(nn.Module):
    """
    A VQVAE model with residual CNNs for encoder and decoder
    """
    def __init__(self, dim, codebook_size, decay, commitment_weight, lfq, configs):
        super().__init__()
        self.d_model = dim
        self.num_layers = configs.model.num_layers

        # enc_layers = get_layers(True, self.num_layers, configs)
        # enc_layers.append(ResidualBlock(dim, dim))

        self.encoder_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            nn.ReLU(),
        )

        if not lfq:
            # Regular vector quantization
            self.vq_layer = VectorQuantize(
                dim=self.d_model,
                codebook_size=codebook_size,
                decay=decay,
                commitment_weight=commitment_weight,
                accept_image_fmap=True
            )

        else:
            self.vq_layer = LFQ(
                dim=self.d_model,
                codebook_size=codebook_size
            )

        # dec_layers = get_layers(False, self.num_layers, configs)
        # dec_layers.append(ResidualBlock(3, 3))

        self.decoder_layers = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),

            ResidualBlock(1,1),
            ResidualBlock(1, 1),
            ResidualBlock(1, 1),
            ResidualBlock(1, 1),
            ResidualBlock(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_vq_only=False):

        if type(x) == dict:
            x = x["input_contact_map"]

        x = self.encoder_layers(x)
        x, indices, commit_loss = self.vq_layer(x)

        if return_vq_only:
            return x, indices, commit_loss

        x = self.decoder_layers(x)

        # Convert results to 0 and 1 using a threshold of 0.5
        x = (x >= 0.5).float()

        return x, indices, commit_loss


def prepare_models(configs, logger, accelerator):
    # from torchsummary import summary

    vqvae = VQVAEResNet(
        dim=configs.model.vector_quantization.dim,
        codebook_size=configs.model.vector_quantization.codebook_size,
        decay=configs.model.vector_quantization.decay,
        commitment_weight=configs.model.vector_quantization.commitment_weight,
        lfq=configs.model.vector_quantization.lfq,
        configs=configs
    )

    # if accelerator.is_main_process:
        # print_trainable_parameters(gvp_vqvae, logger, 'VQ-VAE')

    return vqvae


if __name__ == "__main__":

    # Get configs
    config_path = '../configs/config_vqvae_contact.yaml'
    with open(config_path) as file:
        config_file = yaml.full_load(file)
    configs_contact = Box(config_file)

    # Data loader
    testloader, valid_loader, vis_loader = prepare_dataloaders(configs_contact)

    logger_test = get_dummy_logger()
    accelerator_test = Accelerator()

    # Ensure models gets and returns 1x_x_ tensors
    model = prepare_models(configs_contact, logger_test, accelerator_test)

    for data in tqdm(testloader):
        cmaps = data["input_contact_map"]
        print(cmaps.size())
        x_test, indices_test, commit_loss_test = model(cmaps)
        print(cmaps[0].size(), x_test[0].size())
        #assert cmaps[0].size() == torch.Size([3,32,32])
        #assert x_test[0].size() == torch.Size([3,32,32])

