import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def generate_data(n_samples, n_features, centers, cluster_std):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std,
                      random_state=42)
    return X, y


def load_fashion_mnist_data(batch_size, shuffle):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset = datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def main():
    batch_size = 256
    shuffle = True
    train_loader = load_fashion_mnist_data(batch_size, shuffle)
    return train_loader


if __name__ == "__main__":
    train_loader = load_fashion_mnist_data(batch_size=1, shuffle=False)
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i} of images has shape {images.shape}")
        print(f"Batch {i} of labels has shape {labels.shape}")
        img = images.squeeze().numpy()
        img = cv2.resize(img, (256, 256))

        cv2.imshow('image', img)
        cv2.waitKey(0)
