import argparse
import yaml
import torch
from tqdm.auto import trange
import  load_configs, prepare_saving_dir
import numpy as np
from utils  import *
from torch.utils.data import DataLoader
from data_test import *
from model import SimpleVQAutoEncoder


device = "cuda" if torch.cuda.is_available() else "cpu"
num_codes = 256

def train_loop(model, train_loader, train_iterations=1000, alpha=10):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, _ = next(iterate_dataset(train_loader))
        out, indices, cmt_loss = model(x)
        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()

        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )
    return



def evaluate_loop():
    pass


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Train a deep neural nets.")
    # parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    # args = parser.parse_args()
    # config_path = args.config_path
    #
    # with open(config_path) as file:
    #     config_file = yaml.full_load(file)

    train_data = load_fashion_mnist_data(batch_size=256, shuffle=True)

    # result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
    lr = 3e-4
    epoch = 1000
    num_codes = 256
    seed = 1234

    torch.random.manual_seed(seed)
    model = SimpleVQAutoEncoder(codebook_size=num_codes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loop(model, train_data, train_iterations=epoch)
    print("Train finished!")