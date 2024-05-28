import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vector_quantize_pytorch import VectorQuantize
from tqdm import tqdm

import yaml
from utils.utils import get_dummy_logger
from torch.utils.data import DataLoader
from accelerate import Accelerator
from box import Box


class VQVAE(nn.Module):
    """
    A simple VQVAE models for images
    """
    def __init__(self, dim, codebook_size, decay, commitment_weight, configs):
        super().__init__()
        self.d_model = dim

        self.encoder_layers = nn.Sequential(
            nn.Conv2d(3, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.vq_layer = VectorQuantize(
            dim=self.d_model,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            accept_image_fmap=True
        )

        self.decoder_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(self.d_model, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder_layers(x)
        x, indices, commit_loss = self.vq_layer(x)
        x = self.decoder_layers(x)
        return x, indices, commit_loss


def prepare_models(configs, logger, accelerator):
    # from torchsummary import summary

    vqvae = VQVAE(
        dim=configs.model.vector_quantization.dim,
        codebook_size=configs.model.vector_quantization.codebook_size,
        decay=configs.model.vector_quantization.decay,
        commitment_weight=configs.model.vector_quantization.commitment_weight,
        configs=configs
    )

    # if accelerator.is_main_process:
        # print_trainable_parameters(gvp_vqvae, logger, 'VQ-VAE')

    return vqvae


if __name__ == "__main__":

    # Normalize data to be in the range [-1.0, 1.0]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR dataset
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    # Get configs
    config_path = '../configs/config_cifar.yaml'
    with open(config_path) as file:
        config_file = yaml.full_load(file)
    configs_cifar = Box(config_file)

    logger_test = get_dummy_logger()
    accelerator_test = Accelerator()

    # Ensure models gets and returns 3x32x32 tensors
    model = prepare_models(configs_cifar, logger_test, accelerator_test)
    # model = VQVAE(dim=16, codebook_size=32, decay=0.9, commitment_weight=0.9, accept_image_fmap=True)
    for data in tqdm(testloader):
        images, labels = data
        x_test, indices_test, commit_loss_test = model(images)
        assert images[0].size() == torch.Size([3,32,32])
        assert x_test[0].size() == torch.Size([3,32,32])
