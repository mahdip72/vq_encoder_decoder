from torch.utils.data import Dataset
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

    [transforms.ToTensor()])
    #, transforms.Normalize((0.5,), (0.5,))])

    dataset = datasets.FashionMNIST(
        root="~/data/fashion_mnist", train=True, download=True, transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def load_cifar10_data(train, batch_size, shuffle):
    """
    Load a CIFAR 10 dataloader with the given arguments.
    :param train: (bool) True for train dataset, False for test dataset
    :batch_size: (int) size of each batch
    :shuffle: (bool) whether to shuffle the dataset
    :return: (DataLoader) dataloader for the CIFAR 10 dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
    ])

    dataset = datasets.CIFAR10(
        root="~/data/cifar10", train=train, download=True, transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def prepare_dataloaders(configs):
    # train_dataloader = load_fashion_mnist_data(batch_size=configs.train_settings.batch_size, shuffle=True)
    train_dataloader = load_cifar10_data(train=True, batch_size=configs.train_settings.batch_size, shuffle=True)
    test_dataloader = load_cifar10_data(train=False, batch_size=configs.valid_settings.batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def main():
    batch_size = 256
    shuffle = True
    train_loader = load_fashion_mnist_data(batch_size, shuffle)
    return train_loader


if __name__ == "__main__":
    import yaml
    from utils.utils import load_configs

    config_path = "../configs/config_gvp.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main_configs = load_configs(config_file)

    # train_loader = load_fashion_mnist_data(batch_size=1, shuffle=False)
    data_loader = load_cifar10_data(batch_size=1, shuffle=False)
    for i, (images, labels) in enumerate(data_loader):
        print(f"Batch {i} of images has shape {images.shape}")
        print(f"Batch {i} of labels has shape {labels.shape}")
        img = images.squeeze().numpy()
        # convert CHW to HWC
        img = np.transpose(img, (1, 2, 0))
        # convert to 0-255 based on the normalization values in the transform
        # img = img * 0.5 + 0.5

        img = cv2.resize(img, (256, 256))

        cv2.imshow('image', img[:, :, ::-1])
        cv2.waitKey(0)
