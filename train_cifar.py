import argparse
import numpy as np
import yaml
import torch
from utils.utils import load_configs, prepare_saving_dir, get_logging, prepare_optimizer, prepare_tensorboard, save_checkpoint
from utils.utils import load_checkpoints
from accelerate import Accelerator
from data.data_cifar import prepare_dataloaders
from models.vqvae_cifar import prepare_models
from tqdm import tqdm
import os


def train_loop(model, train_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    train_writer = kwargs.pop('train_writer')
    configs = kwargs.pop('configs')
    alpha = configs.model.vector_quantization.alpha
    codebook_size = configs.model.vector_quantization.codebook_size
    accum_iter = configs.train_settings.grad_accumulation

    optimizer.zero_grad()

    model.train()
    train_loss = 0.0
    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0
    counter = 0
    global_step = kwargs.get('global_step', 0)

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / accum_iter))),
                        leave=False, disable=not accelerator.is_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    # Training loop
    for images, labels in train_loader:

        # Train with gradient accumulation
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            outputs, indices, commit_loss = model(images)

            # Consider both reconstruction loss and commit loss
            rec_loss = torch.nn.functional.l1_loss(images, outputs)
            loss = rec_loss + alpha * commit_loss

            # Gather the losses across all processes for logging (if we use distributed training).
            # TODO: what is the point of train_loss?
            avg_loss = accelerator.gather(loss.repeat(configs.train_settings.batch_size)).mean()
            train_loss += avg_loss.item() / accum_iter

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), configs.optimizer.grad_clip_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            counter += 1

            # Keep track of total combined loss, total reconstruction loss, and total commit loss
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_cmt_loss += commit_loss.item()
            train_loss = 0

            progress_bar.set_description(f"epoch {epoch} "
                                         + f"[loss: {total_loss / counter:.3f}, "
                                         + f"rec loss: {total_rec_loss / counter:.3f}, "
                                         + f"cmt loss: {total_cmt_loss / counter:.3f}]")

            # Add learning rate to TensorBoard for each global step
            if accelerator.is_main_process:
                train_writer.add_scalar('Train/Learning Rate', optimizer.param_groups[0]['lr'], global_step)

        progress_bar.set_postfix(
            {
                "lr": optimizer.param_groups[0]['lr'],
                "step_loss": loss.detach().item(),
                "rec_loss": rec_loss.detach().item(),
                "cmt_loss": commit_loss.detach().item(),
                "activation": indices.unique().numel() / codebook_size * 100,
                "global_step": global_step
            }
        )

    avg_loss = total_loss / counter
    avg_rec_loss = total_rec_loss / counter
    avg_cmt_loss = total_cmt_loss / counter

    return_dict = {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "cmt_loss": avg_cmt_loss,
        "counter": counter,
        "global_step": global_step
    }
    return return_dict


def valid_loop(model, valid_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    valid_writer = kwargs.pop('valid_writer')
    configs = kwargs.pop('configs')
    alpha = configs.model.vector_quantization.alpha
    codebook_size = configs.model.vector_quantization.codebook_size
    accum_iter = configs.train_settings.grad_accumulation

    optimizer.zero_grad()

    valid_loss = 0.0
    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0
    counter = 0
    global_step = kwargs.get('global_step', 0)

    progress_bar = tqdm(range(0, int(len(valid_loader))),
                        leave=False, disable=not accelerator.is_main_process)
    progress_bar.set_description(f"Validation epoch {epoch}")

    # Validation loop
    model.eval()
    for images, labels in valid_loader:

        with torch.inference_mode():
            optimizer.zero_grad()
            outputs, indices, commit_loss = model(images)
            images = images.to(accelerator.device)
            outputs = outputs.to(accelerator.device)

            # Consider both reconstruction loss and commit loss
            rec_loss = torch.nn.functional.l1_loss(images, outputs)
            loss = rec_loss + alpha * commit_loss

        # global_step += 1
        progress_bar.update(1)
        counter += 1

        # Keep track of total combined loss, total reconstruction loss, and total commit loss
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_cmt_loss += commit_loss.item()

        progress_bar.set_description(f"validation epoch {epoch} "
                                     + f"[loss: {total_loss / counter:.3f}, "
                                     + f"rec loss: {total_rec_loss / counter:.3f}, "
                                     + f"cmt loss: {total_cmt_loss / counter:.3f}]")

    avg_loss = total_loss / counter
    avg_rec_loss = total_rec_loss / counter
    avg_cmt_loss = total_cmt_loss / counter

    return_dict = {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "cmt_loss": avg_cmt_loss,
        "counter": counter,
    }
    return return_dict


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

    # Prepare dataloader, model, and optimizer
    train_dataloader, valid_dataloader = prepare_dataloaders(configs)

    if accelerator.is_main_process:
        logging.info('Finished preparing dataloaders')

    net = prepare_models(configs, logging, accelerator)
    if accelerator.is_main_process:
        logging.info('Finished preparing models')

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    if accelerator.is_main_process:
        logging.info('Finished preparing optimizer')

    net, optimizer, train_dataloader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, scheduler
    )

    # Load checkpoints if needed
    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator)

    net.to(accelerator.device)

    # compile models to train faster and efficiently
    if configs.model.compile_model:
        net = torch.compile(net)
        if accelerator.is_main_process:
            logging.info('Finished compiling models')

    # Initialize train and valid TensorBoards
    train_writer, valid_writer = prepare_tensorboard(result_path)

    # Log number of train steps per epoch
    if accelerator.is_main_process:
        train_steps = np.ceil(len(train_dataloader) / configs.train_settings.grad_accumulation)
        logging.info(f'Number of train steps per epoch: {int(train_steps)}')

    # Keep track of global step across all processes; useful for continuing training from a checkpoint.
    global_step=0
    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):

        # Training
        training_loop_reports = train_loop(net, train_dataloader, epoch,
                                                                accelerator=accelerator,
                                                                optimizer=optimizer,
                                                                scheduler=scheduler, configs=configs,
                                                                logging=logging, global_step=global_step,
                                                                train_writer=train_writer)

        if accelerator.is_main_process:
            logging.info(
                f'epoch {epoch} ({training_loop_reports["counter"]} steps) - '
                f'global steps {training_loop_reports["global_step"]}, loss {training_loop_reports["loss"]:.4f}, '
                f'rec loss {training_loop_reports["rec_loss"]:.4f}, '
                f'cmt loss {training_loop_reports["cmt_loss"]:.4f}')

        global_step = training_loop_reports["global_step"]

        # Validation
        valid_loop_reports = valid_loop(net, valid_dataloader, epoch,
                                                                accelerator=accelerator,
                                                                optimizer=optimizer,
                                                                scheduler=scheduler, configs=configs,
                                                                logging=logging, global_step=global_step,
                                                                valid_writer=valid_writer)

        # Save checkpoints
        if epoch % configs.checkpoints_every == 0:
            accelerator.wait_for_everyone()
            # Set the path to save the model's checkpoint.
            model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')

            if accelerator.is_main_process:
                save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler)
                logging.info(f'\tsaving the best models in {model_path}')

        # Add train losses to TensorBoard
        if accelerator.is_main_process:
            train_writer.add_scalar('Train/Combined Loss', training_loop_reports['loss'], epoch)
            train_writer.add_scalar('Train/Reconstruction Loss', training_loop_reports["rec_loss"], epoch)
            train_writer.add_scalar('Train/Commitment Loss', training_loop_reports["cmt_loss"], epoch)
            train_writer.flush()

        # Add validation losses to TensorBoard
        if accelerator.is_main_process:
            valid_writer.add_scalar('Validation/Combined Loss', valid_loop_reports['loss'], epoch)
            valid_writer.add_scalar('Validation/Reconstruction Loss', valid_loop_reports["rec_loss"], epoch)
            valid_writer.add_scalar('Validation/Commitment Loss', valid_loop_reports["cmt_loss"], epoch)
            valid_writer.flush()

    train_writer.close()
    valid_writer.close()

    if accelerator.is_main_process:
        logging.info('Training complete!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a VQ-VAE model.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./configs/config_cifar.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)