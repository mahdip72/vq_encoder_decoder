# 32x32x3 tensors
# Layers: CNN, batch normalization, VQ

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from vector_quantize_pytorch import VectorQuantize, LFQ


class VQVAE(nn.Module):
    """
    A simple VQVAE model for images
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.d_model = kwargs['dim']

        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(3, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(self.d_model),
            nn.GELU()
        ])

        self.vq_layer = VectorQuantize(
            dim=self.d_model,
            codebook_size=kwargs['codebook_size'],
            decay=kwargs['decay'],
            commitment_weight=kwargs['commitment_weight'],
            accept_image_fmap=True
        )

        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1)
        ])


    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        x, indices, commit_loss = self.vq_layer(x)
        return x, indices, commit_loss


if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    model = VQVAE(dim=1, codebook_size=32, decay=0.9, commitment_weight=0.9, accept_image_fmap=True)
    for data in testloader:
        images, labels = data
        quantized, indices, commit_loss = model(images)
        print(indices)
        break