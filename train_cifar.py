import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
from utils.utils import load_configs, prepare_saving_dir, get_logging, prepare_optimizer, prepare_tensorboard, save_checkpoint
from utils.utils import load_checkpoints
from utils.utils import get_dummy_logger
from accelerate import Accelerator
from data.data_cifar import prepare_dataloaders
from models.vqvae_model import prepare_models
from tqdm import tqdm
import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from box import Box


def train_loop(model, train_loader, optimizer, scheduler, epoch, configs, accelerator):

    alpha = configs.model.vector_quantization.alpha
    codebook_size = configs.model.vector_quantization.codebook_size

    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

    for images, labels in pbar:
        loss = torch.Tensor(0)

        # Train with gradient accumulation
        with accelerator.accumulate(model):
            outputs, indices, commit_loss = model(images)

            # Consider both reconstruction loss and commit loss
            rec_loss = torch.nn.functional.l1_loss(images, outputs)
            loss = rec_loss + alpha * commit_loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), configs.optimizer.grad_clip_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Print loss for each epoch
        total_loss += loss.item()
        batch_avg_loss = total_loss / (pbar.n + 1)
        pbar.set_description(
            f"Epoch: {epoch}, Batch Avg Loss: {batch_avg_loss:.3f} | "
            + f"Rec Loss: {rec_loss.item():.3f} | "
            + f"Cmt Loss: {commit_loss.item():.3f} | "
            + f"Active %: {indices.unique().numel() / codebook_size * 100:.3f}")

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def load_configs_cifar(configs):
    """
    Temporary function for loading CIFAR configs
    """
    tree_config = Box(config_file)

    # Convert the necessary values to floats.
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)

    return tree_config


def plot_loss(epochs, loss):
    """
    Make a plot with loss on the y-axis and epochs on the x-axis.
    :param epochs: list of epoch numbers
    :param loss: list of loss values
    """
    plt.plot(epochs, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def main(dict_config, config_file_path):
    configs = load_configs_cifar(dict_config)

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
    logging = get_logging(result_path)

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dispatch_batches=False
    )

    # Prepare dataloader, model, and optimizer
    train_dataloader = prepare_dataloaders(configs)
    logging.info('Finished preparing dataloaders')
    net = prepare_models(configs, logging, accelerator)
    logging.info('Finished preparing models')
    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    logging.info('Finished preparing optimizer')

    net, optimizer, train_dataloader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, scheduler
    )

    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator)

    net.to(accelerator.device)

    # compile models to train faster and efficiently
    if configs.model.compile_model:
        net = torch.compile(net)
        if accelerator.is_main_process:
            logging.info('Finished compiling models')

    """
    # Initialize train and valid TensorBoards
    train_writer, valid_writer = prepare_tensorboard(result_path)
    """

    # Training loop
    loss = []
    epochs = []
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):
        train_loss = train_loop(net, train_dataloader, optimizer, scheduler, epoch, configs, accelerator)
        loss.append(train_loss)
        epochs.append(epoch)

        # Save checkpoints
        if epoch % configs.checkpoints_every == 0:
            tools = dict()
            tools['net'] = net
            tools['optimizer'] = optimizer
            tools['scheduler'] = scheduler

            accelerator.wait_for_everyone()

            # Set the path to save the model's checkpoint.
            model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
            if accelerator.is_main_process:
                save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler)
                logging.info(f'\tsaving the best models in {model_path}')

        if accelerator.is_main_process:
            logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}')

    # Plot loss across all epochs
    plot_loss(epochs, loss)

    logging.info('Training complete!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a VQ-VAE model.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./configs/config_cifar.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)