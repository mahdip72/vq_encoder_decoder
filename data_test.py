import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import numpy as np


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


def main():
    # Set a fixed random seed for reproducibility
    np.random.seed(42)

    # Parameters for dataset generation
    n_samples = 10000
    n_features = 8
    centers = 16
    cluster_std = 1.2
    test_size = 0.2
    batch_size = 64

    # Generate data
    X, y = generate_data(n_samples, n_features, centers, cluster_std)

    # Prepare datasets
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=42)
    train_dataset = CustomDataset(x_train, y_train)
    valid_dataset = CustomDataset(x_valid, y_valid)

    # Prepare DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


if __name__ == "__main__":
    main()
