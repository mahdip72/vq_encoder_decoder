import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
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


if __name__ == '__main__':
    net = SimpleVQAutoEncoder(codebook_size=256)
    # create a random input tensor and pass it through the network
    x = torch.randn(1, 1, 28, 28)
    output, x, y = net(x, return_vq_only=True)
    print(output.shape)
    print(x.shape)
    print(y.shape)
