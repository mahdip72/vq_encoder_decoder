import argparse
import yaml
import torch
from tqdm.auto import trange
import numpy as np
from utils  import *
from torch.utils.data import DataLoader
from data_test import *
from model import SimpleVQAutoEncoder
from tqdm import  tqdm


def train(model, train_loader, optimizer, device, epoch, alpha, num_codes):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for data in pbar:
        inputs, labels = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, indices, cmt_loss = model(inputs)
        rec_loss = torch.abs(outputs - inputs).mean()
        loss = rec_loss + alpha * cmt_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate the average loss for the current batch for real-time feedback
        batch_avg_loss = total_loss / (pbar.n + 1)  # pbar.n is the number of batches processed so far

        # Update progress description with current loss values
        pbar.set_description(
            f"Epoch: {epoch}, Batch Avg Loss: {batch_avg_loss:.3f} | "
            + f"Rec Loss: {rec_loss.item():.3f} | "
            + f"Cmt Loss: {cmt_loss.item():.3f} | "
            + f"Active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )

    # Calculate the average loss across all batches for the epoch
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    epochs = 10
    lr = 3e-4
    alpha = 10  # Regularization factor for commitment loss
    num_codes = 256
    train_data = load_fashion_mnist_data(batch_size=256, shuffle=True)

    model = SimpleVQAutoEncoder(codebook_size=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_data, optimizer, device, epoch, alpha, num_codes)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}')

    print("Training complete!")


if __name__ == '__main__':
    main()
