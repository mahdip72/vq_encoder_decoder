import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vector_quantize_pytorch import VectorQuantize


class VQVAE(nn.Module):
    """
    A simple VQVAE model for images
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.d_model = kwargs['dim']

        self.encoder_layers = nn.Sequential(
            nn.Conv2d(3, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.vq_layer = VectorQuantize(
            dim=self.d_model,
            codebook_size=kwargs['codebook_size'],
            decay=kwargs['decay'],
            commitment_weight=kwargs['commitment_weight'],
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


if __name__ == "__main__":

    # Normalize data to be in the range [-1.0, 1.0]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR dataset
    testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    # Ensure model gets and returns 3x32x32 tensors
    model = VQVAE(dim=16, codebook_size=32, decay=0.9, commitment_weight=0.9, accept_image_fmap=True)
    for data in testloader:
        images, labels = data
        print(images[0].size())
        x_test, indices_test, commit_loss_test = model(images)
        print(x_test[0].shape)

        assert images[0].size() == torch.Size([3,32,32])
        assert x_test[0].size() == torch.Size([3,32,32])
