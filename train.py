import argparse
import numpy as np
import yaml
import os
import torch
from utils import load_configs, prepare_saving_dir, get_logging, prepare_optimizer, prepare_tensorboard, save_checkpoint
from utils import load_checkpoints
from accelerate import Accelerator
from data_test import prepare_dataloaders
from model import prepare_models
from tqdm import tqdm


def train_loop(net, train_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    logging = kwargs.pop('logging')
    configs = kwargs.pop('configs')
    alpha = configs.model.vector_quantization.alpha
    codebook_size = configs.model.vector_quantization.codebook_size

    optimizer.zero_grad()

    net.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for data in pbar:
        inputs, labels = data
        optimizer.zero_grad()
        outputs, indices, cmt_loss = net(inputs)

        rec_loss = torch.nn.functional.l1_loss(outputs, inputs)

        loss = rec_loss + alpha * cmt_loss

        # Gather the losses across all processes for logging (if we use distributed training).
        # avg_loss = accelerator.gather(loss.repeat(kwargs['configs'].train_settings.train_batch_size)).mean()
        # train_loss += avg_loss.item() / kwargs['configs'].train_settings.grad_accumulation

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(net.parameters(), configs.optimizer.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        batch_avg_loss = total_loss / (pbar.n + 1)
        pbar.set_description(
            f"Epoch: {epoch}, Batch Avg Loss: {batch_avg_loss:.3f} | "
            + f"Rec Loss: {rec_loss.item():.3f} | "
            + f"Cmt Loss: {cmt_loss.item():.3f} | "
            + f"Active %: {indices.unique().numel() / codebook_size * 100:.3f}")

    avg_loss = total_loss / len(train_loader)

    return avg_loss


def main(dict_config, config_file_path):
    configs = load_configs(dict_config)

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    logging = get_logging(result_path)

    train_dataloader = prepare_dataloaders(configs)
    logging.info('preparing dataloaders are done')

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dispatch_batches=True
    )

    net = prepare_models(configs, logging, accelerator)
    logging.info('preparing model is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    logging.info('preparing optimizer is done')

    net, optimizer, train_dataloader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, scheduler
    )

    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator)

    net.to(accelerator.device)

    # compile model to train faster and efficiently
    if configs.model.compile_model:
        net = torch.compile(net)
        if accelerator.is_main_process:
            logging.info('compile model is done')

    # Initialize train and valid TensorBoards
    train_writer, valid_writer = prepare_tensorboard(result_path)

    for epoch in range(1, configs.train_settings.num_epochs + 1):
        train_loss = train_loop(net, train_dataloader, epoch,
                                accelerator=accelerator, optimizer=optimizer, scheduler=scheduler, configs=configs,
                                logging=logging, train_writer=train_writer)

        if epoch % configs.checkpoints_every == 0:
            tools = dict()
            tools['net'] = net
            tools['optimizer'] = optimizer
            tools['scheduler'] = scheduler

            accelerator.wait_for_everyone()

            # Set the path to save the model checkpoint.
            model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
            save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler)
            if accelerator.is_main_process:
                logging.info(f'\tsaving the best model in {model_path}')

        logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}')

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-VAE model.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
