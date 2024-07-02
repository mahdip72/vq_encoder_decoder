import argparse
import numpy as np
import yaml
import torch
from torch.nn import BCELoss
from utils.utils import load_configs, prepare_saving_dir, get_logging, prepare_optimizer, prepare_tensorboard, \
    save_checkpoint
from utils.utils import load_checkpoints
from accelerate import Accelerator
from data.data_contactmap import prepare_dataloaders
from models.vqvae_contact import prepare_models
from tqdm import tqdm
import os
import time
import torchmetrics
from visualization.main import compute_visualization


# from ray import tune
# from ray import train
# from ray.tune.schedulers import ASHAScheduler


def train_loop(model, train_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    train_writer = kwargs.pop('train_writer')
    configs = kwargs.pop('configs')
    alpha = configs.model.vector_quantization.alpha
    codebook_size = configs.model.vector_quantization.codebook_size
    accum_iter = configs.train_settings.grad_accumulation

    # Prepare metrics for evaluation
    rmse = torchmetrics.MeanSquaredError(squared=False)
    mae = torchmetrics.MeanAbsoluteError()
    bce = BCELoss()

    rmse.to(accelerator.device)
    mae.to(accelerator.device)
    bce.to(accelerator.device)

    optimizer.zero_grad()

    train_total_loss = 0.0
    train_rec_loss = 0.0
    train_cmt_loss = 0.0

    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0

    total_activation = 0.0
    counter = 0
    global_step = kwargs.get('global_step', 0)

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / accum_iter))),
                        leave=False,
                        disable=not (accelerator.is_main_process and configs.tqdm_progress_bar))
    progress_bar.set_description(f"Epoch {epoch}")

    # Training loop
    model.train()
    for data in train_loader:
        cmaps = data["input_contact_map"]

        # Train with gradient accumulation
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            outputs, indices, commit_loss = model(cmaps)

            # Consider both reconstruction loss and commit loss
            # rec_loss = torch.nn.functional.l1_loss(cmaps, outputs)

            # Binary cross entropy loss
            rec_loss = bce(cmaps.reshape(-1, 1), outputs.reshape(-1, 1))

            loss = rec_loss + alpha * commit_loss

            # Update the metrics
            mae.update(accelerator.gather(cmaps.detach()), accelerator.gather(outputs.detach()))
            rmse.update(accelerator.gather(cmaps.detach()), accelerator.gather(outputs.detach()))

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_rec_loss = accelerator.gather(rec_loss.detach().repeat(configs.train_settings.batch_size)).mean()
            train_rec_loss += avg_rec_loss.item() / accum_iter

            avg_cmt_loss = accelerator.gather(commit_loss.detach().repeat(configs.train_settings.batch_size)).mean()
            train_cmt_loss += avg_cmt_loss.item() / accum_iter

            train_total_loss = train_rec_loss + alpha * train_cmt_loss

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), configs.optimizer.grad_clip_norm)

            optimizer.step()
            scheduler.step()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            counter += 1

            # Keep track of total combined loss, total reconstruction loss, and total commit loss
            total_loss += train_total_loss
            total_rec_loss += train_rec_loss
            total_cmt_loss += train_cmt_loss
            total_activation += indices.unique().numel() / codebook_size

            train_total_loss = 0.0
            train_rec_loss = 0.0
            train_cmt_loss = 0.0

            progress_bar.set_description(f"epoch {epoch} "
                                         + f"rec loss: {total_rec_loss / counter:.3f}, "
                                         + f"cmt loss: {total_cmt_loss / counter:.3f}]")

            # Add learning rate to TensorBoard for each global step
            if accelerator.is_main_process and configs.tensorboard_log:
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

    # Compute average losses and metrics
    avg_loss = total_loss / counter
    avg_rec_loss = total_rec_loss / counter
    avg_cmt_loss = total_cmt_loss / counter
    avg_activation = total_activation / counter

    return_dict = {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "cmt_loss": avg_cmt_loss,
        "rec_mae": mae.compute().item(),
        "rec_rmse": rmse.compute().item(),
        "counter": counter,
        "global_step": global_step
    }

    # Reset the metrics for the next epoch
    mae.reset()
    rmse.reset()

    return return_dict


def valid_loop(model, valid_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    configs = kwargs.pop('configs')
    alpha = configs.model.vector_quantization.alpha
    accum_iter = configs.train_settings.grad_accumulation

    # Prepare metrics to evaluation
    rmse = torchmetrics.MeanSquaredError(squared=False)
    mae = torchmetrics.MeanAbsoluteError()
    bce = BCELoss()

    rmse.to(accelerator.device)
    mae.to(accelerator.device)
    bce.to(accelerator.device)

    optimizer.zero_grad()

    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0

    valid_total_loss = 0.0
    valid_rec_loss = 0.0
    valid_cmt_loss = 0.0
    counter = 0
    global_step = kwargs.get('global_step', 0)

    progress_bar = tqdm(range(0, int(len(valid_loader))),
                        leave=False,
                        disable=not (accelerator.is_main_process and configs.tqdm_progress_bar))
    progress_bar.set_description(f"Validation epoch {epoch}")

    # Validation loop
    model.eval()
    for data in valid_loader:
        cmaps = data["input_contact_map"]

        with torch.inference_mode():
            optimizer.zero_grad()
            outputs, indices, commit_loss = model(cmaps)
            cmaps = cmaps.to(accelerator.device)
            outputs = outputs.to(accelerator.device)

            # Consider both reconstruction loss and commit loss
            # rec_loss = torch.nn.functional.l1_loss(cmaps, outputs)

            # Binary cross entropy loss
            rec_loss = bce(cmaps.reshape(-1, 1), outputs.reshape(-1, 1))

            loss = rec_loss + alpha * commit_loss

            # Update the metrics
            mae.update(accelerator.gather(cmaps).detach(), accelerator.gather(outputs).detach())
            rmse.update(accelerator.gather(cmaps).detach(), accelerator.gather(outputs).detach())

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_rec_loss = accelerator.gather(rec_loss.repeat(configs.valid_settings.batch_size)).mean()
            valid_rec_loss += avg_rec_loss.item() / accum_iter

            avg_cmt_loss = accelerator.gather(commit_loss.repeat(configs.valid_settings.batch_size)).mean()
            valid_cmt_loss += avg_cmt_loss.item() / accum_iter

            valid_total_loss = valid_rec_loss + alpha * valid_cmt_loss

        # global_step += 1
        progress_bar.update(1)
        counter += 1

        # Keep track of total combined loss, total reconstruction loss, and total commit loss
        total_loss += valid_total_loss
        total_rec_loss += valid_rec_loss
        total_cmt_loss += valid_cmt_loss

        valid_total_loss = 0.0
        valid_rec_loss = 0.0
        valid_cmt_loss = 0.0
        valid_ce_loss = 0.0

        progress_bar.set_description(f"validation epoch {epoch} "
                                     + f"rec loss: {total_rec_loss / counter:.3f}, "
                                     + f"cmt loss: {total_cmt_loss / counter:.3f}]")

    avg_loss = total_loss / counter
    avg_rec_loss = total_rec_loss / counter
    avg_cmt_loss = total_cmt_loss / counter

    return_dict = {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "cmt_loss": avg_cmt_loss,
        "rec_mae": mae.compute().item(),
        "rec_rmse": rmse.compute().item(),
        "counter": counter,
    }

    # Reset the metrics for the next epoch
    mae.reset()
    rmse.reset()

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
        # gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        # dispatch_batches=False
    )

    # Prepare dataloader, model, and optimizer
    train_dataloader, valid_dataloader, visualization_loader = prepare_dataloaders(configs)

    if accelerator.is_main_process:
        logging.info('Finished preparing dataloaders')

    net = prepare_models(configs, logging, accelerator)
    if accelerator.is_main_process:
        logging.info('Finished preparing models')

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    if accelerator.is_main_process:
        logging.info('Finished preparing optimizer')

    net, optimizer, train_dataloader, valid_dataloader, visualization_loader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, valid_dataloader, visualization_loader, scheduler
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
    train_writer, valid_writer = None, None
    if configs.tensorboard_log:
        train_writer, valid_writer = prepare_tensorboard(result_path)

    # Log number of train steps per epoch
    if accelerator.is_main_process:
        train_steps = np.ceil(len(train_dataloader) / configs.train_settings.grad_accumulation)
        logging.info(f'Number of train steps per epoch: {int(train_steps)}')

    # Keep track of global step across all processes; useful for continuing training from a checkpoint.
    global_step = 0
    # Keep track of best metrics
    best_valid_metrics = {'loss': float('inf'), 'mae': 0.0, 'rmse': 0.0}
    training_loop_reports = dict()
    valid_loop_reports = dict()

    for epoch in range(start_epoch, configs.train_settings.num_epochs + 1):

        # Training
        start_time = time.time()
        training_loop_reports = train_loop(net, train_dataloader, epoch,
                                           accelerator=accelerator,
                                           optimizer=optimizer,
                                           scheduler=scheduler, configs=configs,
                                           logging=logging, global_step=global_step,
                                           train_writer=train_writer)
        end_time = time.time()
        training_time = end_time - start_time
        torch.cuda.empty_cache()

        if accelerator.is_main_process:
            logging.info(
                f'epoch {epoch} ({training_loop_reports["counter"]} steps) - time {np.round(training_time, 2)}, '
                f'global steps {training_loop_reports["global_step"]}, loss {training_loop_reports["loss"]:.4f}, '
                f'rec loss {training_loop_reports["rec_loss"]:.4f}, '
                f'cmt loss {training_loop_reports["cmt_loss"]:.4f}, '
                f'rec mae {training_loop_reports["rec_mae"]:.4f}, '
                f'rec rmse {training_loop_reports["rec_rmse"]:.4f}'
            )

        global_step = training_loop_reports["global_step"]

        # Add train losses to TensorBoard
        if accelerator.is_main_process and configs.tensorboard_log:
            train_writer.add_scalar('Combined Loss', training_loop_reports['loss'], epoch)
            train_writer.add_scalar('Reconstruction Loss', training_loop_reports["rec_loss"], epoch)
            train_writer.add_scalar('Commitment Loss', training_loop_reports["cmt_loss"], epoch)
            train_writer.add_scalar('Reconstruction MAE', training_loop_reports["rec_mae"], epoch)
            train_writer.add_scalar('Reconstruction RMSE', training_loop_reports["rec_rmse"], epoch)
            train_writer.flush()

        # Validation
        if epoch % configs.valid_settings.do_every == 0:
            start_time = time.time()
            valid_loop_reports = valid_loop(net, valid_dataloader, epoch,
                                            accelerator=accelerator,
                                            optimizer=optimizer,
                                            scheduler=scheduler, configs=configs,
                                            logging=logging, global_step=global_step,
                                            valid_writer=valid_writer)
            end_time = time.time()
            valid_time = end_time - start_time
            if accelerator.is_main_process:
                logging.info(
                    f'validation epoch {epoch} ({valid_loop_reports["counter"]} steps) - time {np.round(valid_time, 2)}s, '
                    f'loss {valid_loop_reports["loss"]:.4f}, '
                    f'rec loss {valid_loop_reports["rec_loss"]:.4f}, '
                    f'rec mae {training_loop_reports["rec_mae"]:.4f}, '
                    f'rec rmse {training_loop_reports["rec_rmse"]:.4f}'
                )

            # Add validation losses to TensorBoard
            valid_loss = valid_loop_reports['loss']
            if accelerator.is_main_process and configs.tensorboard_log:
                valid_writer.add_scalar('Combined Loss', valid_loss, epoch)
                valid_writer.add_scalar('Reconstruction Loss', valid_loop_reports["rec_loss"], epoch)
                valid_writer.add_scalar('Commitment Loss', valid_loop_reports["cmt_loss"], epoch)
                valid_writer.add_scalar('Reconstruction MAE', valid_loop_reports["rec_mae"], epoch)
                valid_writer.add_scalar('Reconstruction RMSE', valid_loop_reports["rec_rmse"], epoch)
                valid_writer.flush()

            # Save checkpoints only if current validation loss is less than previous minimum validation loss
            if epoch % configs.checkpoints_every == 0:

                if valid_loss < best_valid_metrics['loss']:
                    best_valid_metrics['loss'] = valid_loss
                    best_valid_metrics['mae'] = valid_loop_reports['rec_mae']
                    best_valid_metrics['rmse'] = valid_loop_reports['rec_rmse']

                    accelerator.wait_for_everyone()
                    # Set the path to save the model's checkpoint.
                    model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')

                    if accelerator.is_main_process:
                        save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer,
                                        scheduler=scheduler)
                        logging.info(f'\tsaving the best models in {model_path}')

                else:
                    if accelerator.is_main_process:
                        logging.info(f'\tvalidation loss higher than previous minimum; did not save model')

        if epoch % configs.visualization_settings.do_every == 0:
            if accelerator.is_main_process:
                logging.info(f'\tstart visualization at epoch {epoch}')

            accelerator.wait_for_everyone()
            # Visualize the embeddings using T-SNE
            if accelerator.is_main_process:
                compute_visualization(net, visualization_loader, result_path, configs, logging, accelerator, epoch)

    train_writer.close()
    valid_writer.close()

    if accelerator.is_main_process:
        logging.info('Training complete!')

        # Log best metrics
        logging.info(f"best valid loss: {best_valid_metrics['loss']:.4f}")
        logging.info(f"best valid mae: {best_valid_metrics['mae']:.4f}")
        logging.info(f"best valid rmse: {best_valid_metrics['rmse']:.4f}")

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()
    torch.cuda.empty_cache()

    return training_loop_reports, valid_loop_reports


# def run_ray_tune(dict_config, config_file_path):
#
#     # Save configs to yaml file
#     with open(config_file_path, 'w') as output_file:
#         yaml.dump(dict_config, output_file, default_flow_style=False)
#
#     training_loop_reports, valid_loop_reports = main(dict_config, config_file_path)
#     train.report({
#         "train_rec_loss": training_loop_reports["rec_loss"],
#         "val_rec_loss": valid_loop_reports["rec_loss"]
#     })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQ-VAE model.")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config_vqvae_contact.yaml')

    # ray_tune flag for tuning hyperparameters
    parser.add_argument("--ray_tune", action='store_true')

    args = parser.parse_args()
    config_path = args.config_path
    ray_tune = args.ray_tune

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    # if ray_tune:
    #
    #     # Set hyperparameters to tune
    #     config_file["train_settings"]["batch_size"] = tune.choice([32, 64, 128])
    #     config_file["optimizer"]["lr"] = tune.choice([0.01, 0.001, 0.0001])
    #     config_file["num_layers"] = tune.choice([1,2,4,8])
    #     config_file["model"]["encoder"]["dim"] = tune.choice([4,8,12])
    #     config_file["model"]["decoder"]["dim"] = tune.choice([4,8,12])
    #
    #     # Scheduler for Ray Tune
    #     ray_scheduler = ASHAScheduler(
    #         metric="val_rec_loss",
    #         mode="min",
    #         max_t=config_file["train_settings"]["num_epochs"],
    #         grace_period=1,
    #         reduction_factor=2,
    #     )
    #
    #     tuner = tune.Tuner(
    #         tune.with_parameters(run_ray_tune, config_file_path=config_path),
    #         tune_config=tune.TuneConfig(
    #             scheduler=ray_scheduler,
    #             num_samples=8
    #         ),
    #         run_config=train.RunConfig(storage_path="~/vq_encoder_decoder/results/ray_tune"),
    #         param_space=config_file
    #     )
    #
    #     results = tuner.fit()
    #
    # else:
    main(config_file, config_path)
