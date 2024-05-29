import argparse
import numpy as np
import yaml
import os
import torch
from utils.utils import load_configs, prepare_saving_dir, get_logging, prepare_optimizer, prepare_tensorboard, \
    save_checkpoint
from utils.utils import load_checkpoints
from accelerate import Accelerator
# from data.data_cifar import prepare_dataloaders
from data.dataset import prepare_dataloaders
# from models.models import prepare_models
from models.gvp_vqvae import prepare_models
from tqdm import tqdm


def train_loop(net, train_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    logging = kwargs.pop('logging')
    configs = kwargs.pop('configs')
    alpha = configs.model.vqvae.vector_quantization.alpha
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size

    optimizer.zero_grad()

    net.train()
    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for data in pbar:
        labels = data['coords']
        masks = data['masks']
        optimizer.zero_grad()
        outputs, indices, commit_loss = net(data)

        masked_outputs = outputs[masks]
        masked_labels = labels[masks]
        rec_loss = torch.nn.functional.l1_loss(masked_outputs, masked_labels)

        loss = rec_loss + alpha * commit_loss

        # Gather the losses across all processes for logging (if we use distributed training).
        # avg_loss = accelerator.gather(loss.repeat(kwargs['configs'].train_settings.train_batch_size)).mean()
        # train_loss += avg_loss.item() / kwargs['configs'].train_settings.grad_accumulation

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(net.parameters(), configs.optimizer.grad_clip_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Keep track of total combined loss, total reconstruction loss, and total commit loss
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_cmt_loss += commit_loss.item()
        batch_avg_loss = total_loss / (pbar.n + 1)

        pbar.set_description(
            f"Epoch: {epoch}, Batch Avg Loss: {batch_avg_loss:.3f} | "
            + f"Rec Loss: {rec_loss.item():.3f} | "
            + f"Cmt Loss: {commit_loss.item():.3f} | "
            + f"Active %: {indices.unique().numel() / codebook_size * 100:.3f}")

    avg_loss = total_loss / len(train_loader)
    avg_rec_loss = total_rec_loss / len(train_loader)
    avg_cmt_loss = total_cmt_loss / len(train_loader)

    return avg_loss, avg_rec_loss, avg_cmt_loss


def main(dict_config, config_file_path):
    configs = load_configs(dict_config)

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

    train_dataloader = prepare_dataloaders(logging, accelerator, configs)
    logging.info('preparing dataloaders are done')

    net = prepare_models(configs, logging, accelerator)
    logging.info('preparing models is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    logging.info('preparing optimizer is done')

    net, optimizer, train_dataloader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, scheduler
    )

    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator)

    net.to(accelerator.device)

    # compile models to train faster and efficiently
    if configs.model.compile_model:
        net = torch.compile(net)
        if accelerator.is_main_process:
            logging.info('compile models is done')

    # Initialize train and valid TensorBoards
    train_writer, valid_writer = prepare_tensorboard(result_path)

    for epoch in range(1, configs.train_settings.num_epochs + 1):
        train_loss, train_rec_loss, train_cmt_loss = train_loop(net, train_dataloader, epoch,
                                                                accelerator=accelerator, optimizer=optimizer,
                                                                scheduler=scheduler, configs=configs,
                                                                logging=logging, train_writer=train_writer)

        if epoch % configs.checkpoints_every == 0:
            tools = dict()
            tools['net'] = net
            tools['optimizer'] = optimizer
            tools['scheduler'] = scheduler

            accelerator.wait_for_everyone()

            # Set the path to save the models checkpoint.
            model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
            save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler)
            if accelerator.is_main_process:
                logging.info(f'\tsaving the best models in {model_path}')

        if accelerator.is_main_process:
            logging.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Rec Loss: {train_rec_loss:.4f}, Cmt Loss: {train_cmt_loss:.4f}')

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-VAE models.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./configs/config_gvp.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
