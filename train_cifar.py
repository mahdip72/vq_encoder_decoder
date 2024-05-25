import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.vqvae_model import VQVAE


def train(model, criterion, train_loader, optimizer, epochs):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    # Normalize data to be in the range [-1.0, 1.0]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR dataset
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)