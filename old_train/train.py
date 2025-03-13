import argparse
import numpy as np
import yaml
import os
import torch
from utils.utils import load_configs, load_configs_gvp, prepare_saving_dir, get_logging, prepare_optimizer, \
    prepare_tensorboard, \
    save_checkpoint
from utils.utils import load_checkpoints
from utils.metrics import GDTTS, LDDT
from accelerate import Accelerator
from visualization.main import compute_visualization
from data.normalizer import Protein3DProcessing
from tqdm import tqdm
import time
import torchmetrics
from utils.custom_losses import orientation_loss, bond_angles_loss, bond_lengths_loss, distance_map_loss
from utils.custom_losses import MultiTaskLossWrapper
import gc


def train_loop(net, train_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    configs = kwargs.pop('configs')
    optimizer_name = configs.optimizer.name
    writer = kwargs.pop('writer')
    loss_wrapper = kwargs.pop('loss_wrapper')
    alpha = configs.model.vqvae.vector_quantization.alpha
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    accum_iter = configs.train_settings.grad_accumulation
    auxilary_coeff = configs.auxiliary_loss.coefficient

    # Prepare metrics for evaluation
    rmse = torchmetrics.MeanSquaredError(squared=False)
    mae = torchmetrics.MeanAbsoluteError()
    gdtts = GDTTS()

    rmse.to(accelerator.device)
    mae.to(accelerator.device)
    gdtts.to(accelerator.device)

    # Prepare the normalizer for denormalization
    processor = Protein3DProcessing()
    processor.load_normalizer(configs.normalizer_path)

    optimizer.zero_grad()

    train_total_loss = 0.0
    train_rec_loss = 0.0
    train_cmt_loss = 0.0

    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0
    total_auxilary_loss = 0.0
    total_orientation_loss = 0.0
    total_bond_angles_loss = 0.0
    total_bond_lengths_loss = 0.0
    total_distance_map_loss = 0.0
    total_activation = 0.0
    counter = 0
    global_step = kwargs.get('global_step', 0)

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / accum_iter))),
                        leave=False, disable=not configs.tqdm_progress_bar)
    progress_bar.set_description(f"Epoch {epoch}")

    net.train()
    if optimizer_name == 'schedulerfree':
        optimizer.train()
    for i, data in enumerate(train_loader):
        with accelerator.accumulate(net):
            labels = data['target_coords']
            masks = data['masks']
            optimizer.zero_grad()
            outputs, indices, commit_loss = net(data)

            # Compute the loss
            masked_outputs = outputs[masks]
            masked_labels = labels[masks]
            rec_loss = torch.nn.functional.l1_loss(masked_outputs, masked_labels)

            loss = rec_loss + alpha * commit_loss

            # Compute the auxiliary losses
            auxilary_loss_list = []

            orientation_loss_value = orientation_loss(masked_outputs, masked_labels)
            if configs.auxiliary_loss.orientation:
                auxilary_loss_list.append(orientation_loss_value)
            else:
                orientation_loss_value = orientation_loss_value.detach()

            bond_angles_loss_value = bond_angles_loss(outputs.reshape(-1, labels.shape[1], 4, 3),
                                                      labels.reshape(-1, labels.shape[1], 4, 3), masks)
            if configs.auxiliary_loss.bond_angles:
                auxilary_loss_list.append(bond_angles_loss_value)
            else:
                bond_angles_loss_value = bond_angles_loss_value.detach()

            bond_lengths_loss_value = bond_lengths_loss(outputs.reshape(-1, labels.shape[1], 4, 3),
                                                        labels.reshape(-1, labels.shape[1], 4, 3), masks)
            if configs.auxiliary_loss.bond_lengths:
                auxilary_loss_list.append(bond_lengths_loss_value)
            else:
                bond_lengths_loss_value = bond_lengths_loss_value.detach()

            distance_map_loss_value = distance_map_loss(outputs, labels)
            if configs.auxiliary_loss.distance_map:
                auxilary_loss_list.append(distance_map_loss_value)
            else:
                distance_map_loss_value = distance_map_loss_value.detach()

            if len(auxilary_loss_list) > 0:
                # Combine auxiliary losses using the loss wrapper
                auxilary_loss = loss_wrapper(auxilary_loss_list) * auxilary_coeff
                loss += auxilary_loss
            else:
                auxilary_loss = 0

            # Denormalize the outputs and labels
            masked_outputs = processor.denormalize_coords(masked_outputs.reshape(-1, 4, 3)).reshape(-1, 3)
            masked_labels = processor.denormalize_coords(masked_labels.reshape(-1, 4, 3)).reshape(-1, 3)

            # Update the metrics
            mae.update(accelerator.gather(masked_outputs.detach()), accelerator.gather(masked_labels.detach()))
            rmse.update(accelerator.gather(masked_outputs.detach()), accelerator.gather(masked_labels.detach()))
            gdtts.update(accelerator.gather(masked_outputs.detach()), accelerator.gather(masked_labels.detach()))

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_rec_loss = accelerator.gather(rec_loss.detach().repeat(configs.train_settings.batch_size)).mean()
            train_rec_loss += avg_rec_loss.item() / accum_iter

            avg_cmt_loss = accelerator.gather(commit_loss.detach().repeat(configs.train_settings.batch_size)).mean()
            train_cmt_loss += avg_cmt_loss.item() / accum_iter

            train_total_loss = train_rec_loss + alpha * train_cmt_loss

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if optimizer_name != 'schedulerfree':
                    accelerator.clip_grad_norm_(net.parameters(), configs.optimizer.grad_clip_norm)

            if optimizer_name != 'schedulerfree':
                optimizer.step()
                scheduler.step()
            else:
                optimizer.step()

        if accelerator.sync_gradients:
            if configs.tqdm_progress_bar:
                progress_bar.update(1)

            global_step += 1
            counter += 1

            # Keep track of total combined loss, total reconstruction loss, and total commit loss
            total_loss += train_total_loss
            total_rec_loss += train_rec_loss
            total_cmt_loss += train_cmt_loss
            total_auxilary_loss += auxilary_loss
            total_orientation_loss += orientation_loss_value
            total_bond_angles_loss += bond_angles_loss_value
            total_bond_lengths_loss += bond_lengths_loss_value
            total_distance_map_loss += distance_map_loss_value
            total_activation += indices.unique().numel() / codebook_size

            train_total_loss = 0.0
            train_rec_loss = 0.0
            train_cmt_loss = 0.0

            if configs.tensorboard_log:
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

            if configs.tqdm_progress_bar:
                progress_bar.set_description(f"epoch {epoch} "
                                             + f"[loss: {total_loss / counter:.3f}, "
                                             + f"rec loss: {total_rec_loss / counter:.3f}, "
                                             + f"cmt loss: {total_cmt_loss / counter:.3f}]")
        if configs.tqdm_progress_bar:
            progress_bar.set_postfix(
                {
                    "lr": optimizer.param_groups[0]['lr'],
                    "step_loss": loss.detach().item(),
                    "rec_loss": rec_loss.detach().item(),
                    "cmt_loss": commit_loss.detach().item(),
                    "auxilary_loss": auxilary_loss.detach().item() if auxilary_loss > 0 else 0,
                    "activation": indices.unique().numel() / codebook_size * 100,
                    "global_step": global_step
                }
            )
        # Clear unused variables to avoid memory leakages
        del data, outputs, indices, commit_loss, rec_loss, loss, avg_rec_loss, avg_cmt_loss
        del auxilary_loss, auxilary_loss_list, masked_outputs, masked_labels, orientation_loss_value
        del bond_angles_loss_value, bond_lengths_loss_value, distance_map_loss_value

    # Compute average losses and metrics
    avg_loss = total_loss / counter
    avg_rec_loss = total_rec_loss / counter
    avg_auxilary_loss = total_auxilary_loss / counter
    avg_orientation_loss = total_orientation_loss / counter
    avg_bond_angles_loss = total_bond_angles_loss / counter
    avg_bond_lengths_loss = total_bond_lengths_loss / counter
    avg_distance_map_loss = total_distance_map_loss / counter
    denormalized_rec_mae = mae.compute().cpu().item()
    denormalized_rec_rmse = rmse.compute().cpu().item()
    gdtts_score = gdtts.compute().cpu().item()
    avg_cmt_loss = total_cmt_loss / counter
    avg_activation = total_activation / counter

    # Log the metrics to TensorBoard
    if configs.tensorboard_log:
        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('rec_loss', avg_rec_loss, epoch)
        writer.add_scalar('real_mae', denormalized_rec_mae, epoch)
        writer.add_scalar('real_rmse', denormalized_rec_rmse, epoch)
        writer.add_scalar('gdtts', gdtts_score, epoch)
        writer.add_scalar('cmt_loss', avg_cmt_loss, epoch)
        writer.add_scalar('codebook_activation', np.round(avg_activation, 2), epoch)
        writer.flush()

    # Reset the metrics for the next epoch
    mae.reset()
    rmse.reset()
    gdtts.reset()

    return_dict = {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "auxilary_loss": avg_auxilary_loss,
        "orientation_loss": avg_orientation_loss,
        "bond_angles_loss": avg_bond_angles_loss,
        "bond_lengths_loss": avg_bond_lengths_loss,
        "distance_map_loss": avg_distance_map_loss,
        "denormalized_rec_mae": denormalized_rec_mae,
        "denormalized_rec_rmse": denormalized_rec_rmse,
        "gdtts": gdtts_score,
        "cmt_loss": avg_cmt_loss,
        "counter": counter,
        "global_step": global_step
    }

    accelerator.wait_for_everyone()
    del progress_bar
    torch.cuda.empty_cache()
    gc.collect()

    return return_dict


def valid_loop(net, valid_loader, epoch, **kwargs):
    optimizer = kwargs.pop('optimizer')
    configs = kwargs.pop('configs')
    optimizer_name = configs.optimizer.name
    accelerator = kwargs.pop('accelerator')
    writer = kwargs.pop('writer')
    alpha = configs.model.vqvae.vector_quantization.alpha

    # Prepare metrics to evaluation
    rmse = torchmetrics.MeanSquaredError(squared=False)
    mae = torchmetrics.MeanAbsoluteError()
    gdtts = GDTTS()
    # lddt = LDDT()

    rmse.to(accelerator.device)
    mae.to(accelerator.device)
    gdtts.to(accelerator.device)
    # lddt.to(accelerator.device)

    # Prepare the normalizer for denormalization
    processor = Protein3DProcessing()
    processor.load_normalizer(configs.normalizer_path)

    optimizer.zero_grad()

    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0
    total_orientation_loss = 0.0
    total_bond_angles_loss = 0.0
    total_bond_lengths_loss = 0.0
    total_distance_map_loss = 0.0
    counter = 0

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(range(0, int(len(valid_loader))),
                        leave=False, disable=not configs.tqdm_progress_bar)
    progress_bar.set_description(f"Validation epoch {epoch}")

    net.eval()
    if optimizer_name != 'schedulerfree':
        optimizer.eval()
    for i, data in enumerate(valid_loader):
        with torch.inference_mode():
            labels = data['target_coords']
            masks = data['masks']
            optimizer.zero_grad()
            outputs, indices, commit_loss = net(data)

            # Compute the loss
            masked_outputs = outputs[masks]
            masked_labels = labels[masks]
            rec_loss = torch.nn.functional.l1_loss(masked_outputs, masked_labels)
            orientation_loss_value = orientation_loss(masked_outputs, masked_labels).detach()
            bond_angles_loss_value = bond_angles_loss(outputs.reshape(-1, labels.shape[1], 4, 3),
                                                      labels.reshape(-1, labels.shape[1], 4, 3), masks).detach()
            bond_lengths_loss_value = bond_lengths_loss(outputs.reshape(-1, labels.shape[1], 4, 3),
                                                        labels.reshape(-1, labels.shape[1], 4, 3), masks).detach()
            distance_map_loss_value = distance_map_loss(outputs, labels).detach()
            loss = rec_loss + alpha * commit_loss

            # Denormalize the outputs and labels
            masked_outputs = processor.denormalize_coords(masked_outputs.reshape(-1, 4, 3)).reshape(-1, 3)
            masked_labels = processor.denormalize_coords(masked_labels.reshape(-1, 4, 3)).reshape(-1, 3)

            # Update the metrics
            mae.update(accelerator.gather(masked_outputs.detach()), accelerator.gather(masked_labels.detach()))
            rmse.update(accelerator.gather(masked_outputs.detach()), accelerator.gather(masked_labels.detach()))
            gdtts.update(accelerator.gather(masked_outputs.detach()), accelerator.gather(masked_labels.detach()))
            # lddt.update(accelerator.gather(masked_outputs.detach(), accelerator.gather(masked_labels.detach())

        if configs.tqdm_progress_bar:
            progress_bar.update(1)
        counter += 1

        # Keep track of total combined loss, total reconstruction loss, and total commit loss
        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_cmt_loss += commit_loss.item()
        total_orientation_loss += orientation_loss_value
        total_bond_angles_loss += bond_angles_loss_value
        total_bond_lengths_loss += bond_lengths_loss_value
        total_distance_map_loss += distance_map_loss_value

        if configs.tqdm_progress_bar:
            progress_bar.set_description(f"validation epoch {epoch} "
                                         + f"[loss: {total_loss / counter:.3f}, "
                                         + f"rec loss: {total_rec_loss / counter:.3f}, "
                                         + f"cmt loss: {total_cmt_loss / counter:.3f}]")

    # Compute average losses and metrics
    avg_loss = total_loss / counter
    avg_rec_loss = total_rec_loss / counter
    avg_orientation_loss = total_orientation_loss / counter
    avg_bond_angles_loss = total_bond_angles_loss / counter
    avg_bond_lengths_loss = total_bond_lengths_loss / counter
    avg_distance_map_loss = total_distance_map_loss / counter
    denormalized_rec_mae = mae.compute().cpu().item()
    denormalized_rec_rmse = rmse.compute().cpu().item()
    gdtts_score = gdtts.compute().cpu().item()
    # lddt_score = lddt.compute().cpu().item()

    # Log the metrics to TensorBoard
    if configs.tensorboard_log:
        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('rec_loss', avg_rec_loss, epoch)
        writer.add_scalar('real_mae', denormalized_rec_mae, epoch)
        writer.add_scalar('real_rmse', denormalized_rec_rmse, epoch)
        writer.add_scalar('gdtts', gdtts_score, epoch)
        # writer.add_scalar('val_lddt', lddt_score, epoch)
        writer.flush()

    # Reset the metrics for the next epoch
    mae.reset()
    rmse.reset()
    gdtts.reset()
    # lddt.reset()

    return_dict = {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "orientation_loss": avg_orientation_loss,
        "bond_angles_loss": avg_bond_angles_loss,
        "bond_lengths_loss": avg_bond_lengths_loss,
        "distance_map_loss": avg_distance_map_loss,
        "denormalized_rec_mae": denormalized_rec_mae,
        "denormalized_rec_rmse": denormalized_rec_rmse,
        "gdtts": gdtts_score,
        # "lddt": lddt_score,
        "counter": counter,
    }

    accelerator.wait_for_everyone()
    del progress_bar
    torch.cuda.empty_cache()
    gc.collect()

    return return_dict


def main(dict_config, config_file_path):
    if dict_config["model"]["architecture"] == 'gvp_vqvae':
        configs = load_configs_gvp(dict_config)
    else:
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
    )

    if configs.model.architecture == 'gvp_vqvae':
        from data.dataset import prepare_gvp_vqvae_dataloaders
        train_dataloader, valid_dataloader, visualization_loader = prepare_gvp_vqvae_dataloaders(logging, accelerator,
                                                                                                 configs)
    elif configs.model.architecture == 'distance_map_vqvae':
        from data.dataset import prepare_distance_map_vqvae_dataloaders
        train_dataloader, valid_dataloader, visualization_loader = prepare_distance_map_vqvae_dataloaders(logging,
                                                                                                          accelerator,
                                                                                                          configs)
    else:
        from data.dataset import prepare_egnn_vqvae_dataloaders
        train_dataloader, valid_dataloader, visualization_loader = prepare_egnn_vqvae_dataloaders(logging, accelerator,
                                                                                                  configs)

    logging.info('preparing dataloaders are done')

    if configs.model.architecture == 'vqvae':
        from models.vqvae import prepare_models_vqvae
        net = prepare_models_vqvae(configs, logging, accelerator)
    else:
        raise ValueError(f'Invalid model architecture: {configs.model.architecture}')
    logging.info('preparing models is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    logging.info('preparing optimizer is done')

    net, optimizer, train_dataloader, valid_dataloader, visualization_loader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, valid_dataloader, visualization_loader, scheduler
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

    if accelerator.is_main_process:
        train_steps = np.ceil(len(train_dataloader) / configs.train_settings.grad_accumulation)
        logging.info(f'number of train steps per epoch: {int(train_steps)}')

    num_losses = sum(1 for loss in list(configs.auxiliary_loss.values())[1:] if loss is True)
    # Combine the losses using adaptive weighting
    loss_wrapper = MultiTaskLossWrapper(num_losses=num_losses)

    loss_wrapper = accelerator.prepare([loss_wrapper])[0]

    # Use this to keep track of the global step across all processes.
    # This is useful for continuing training from a checkpoint.
    global_step = 0
    best_valid_metrics = {'gdtts': 0.0, 'mae': 0.0, 'rmse': 0.0, 'lddt': 0.0, 'loss': 1000.0}
    for epoch in range(1, configs.train_settings.num_epochs + 1):
        start_time = time.time()
        training_loop_reports = train_loop(net, train_dataloader, epoch,
                                           accelerator=accelerator,
                                           optimizer=optimizer,
                                           scheduler=scheduler, configs=configs,
                                           logging=logging, global_step=global_step,
                                           writer=train_writer, loss_wrapper=loss_wrapper)
        end_time = time.time()
        training_time = end_time - start_time
        torch.cuda.empty_cache()
        gc.collect()

        if accelerator.is_main_process:
            logging.info(
                f'epoch {epoch} ({training_loop_reports["counter"]} steps) - time {np.round(training_time, 2)}s, '
                f'global steps {training_loop_reports["global_step"]}, loss {training_loop_reports["loss"]:.4f}, '
                f'rec loss {training_loop_reports["rec_loss"]:.4f}, '
                f'auxilary loss {training_loop_reports["auxilary_loss"]:.4f}, '
                f'orientation loss {training_loop_reports["orientation_loss"]:.4f}, '
                f'bond angles loss {training_loop_reports["bond_angles_loss"]:.4f}, '
                f'bond lengths loss {training_loop_reports["bond_lengths_loss"]:.4f}, '
                f'distance map loss {training_loop_reports["distance_map_loss"]:.4f}, '
                f'denormalized rec mae {training_loop_reports["denormalized_rec_mae"]:.4f}, '
                f'denormalized rec rmse {training_loop_reports["denormalized_rec_rmse"]:.4f}, '
                f'gdtts {training_loop_reports["gdtts"]:.4f}, '
                f'cmt loss {training_loop_reports["cmt_loss"]:.4f}')

        global_step = training_loop_reports["global_step"]

        if epoch % configs.checkpoints_every == 0:
            tools = dict()
            tools['net'] = net
            tools['optimizer'] = optimizer
            tools['scheduler'] = scheduler

            accelerator.wait_for_everyone()

            # Set the path to save the models checkpoint.
            model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
            save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler,
                            configs=configs)
            if accelerator.is_main_process:
                logging.info(f'\tcheckpoint models in {model_path}')

        if epoch % configs.valid_settings.do_every == 0:
            start_time = time.time()
            valid_loop_reports = valid_loop(net, valid_dataloader, epoch,
                                            accelerator=accelerator,
                                            optimizer=optimizer,
                                            scheduler=scheduler, configs=configs,
                                            logging=logging, global_step=global_step,
                                            writer=valid_writer, loss_wrapper=loss_wrapper)
            end_time = time.time()
            valid_time = end_time - start_time
            if accelerator.is_main_process:
                logging.info(
                    f'validation epoch {epoch} ({valid_loop_reports["counter"]} steps) - time {np.round(valid_time, 2)}s, '
                    f'loss {valid_loop_reports["loss"]:.4f}, '
                    f'rec loss {valid_loop_reports["rec_loss"]:.4f}, '
                    f'orientation loss {valid_loop_reports["orientation_loss"]:.4f}, '
                    f'bond angles loss {valid_loop_reports["bond_angles_loss"]:.4f}, '
                    f'bond lengths loss {valid_loop_reports["bond_lengths_loss"]:.4f}, '
                    f'distance map loss {valid_loop_reports["distance_map_loss"]:.4f}, '
                    f'denormalized rec mae {valid_loop_reports["denormalized_rec_mae"]:.4f}, '
                    f'denormalized rec rmse {valid_loop_reports["denormalized_rec_rmse"]:.4f}, '
                    f'gdtts {valid_loop_reports["gdtts"]:.4f}'
                    # f'lddt {valid_loop_reports["lddt"]:.4f}'
                )

            # Check valid metric to save the best model
            if valid_loop_reports["gdtts"] > best_valid_metrics['gdtts']:
                best_valid_metrics['gdtts'] = valid_loop_reports["gdtts"]
                best_valid_metrics['mae'] = valid_loop_reports["denormalized_rec_mae"]
                best_valid_metrics['rmse'] = valid_loop_reports["denormalized_rec_rmse"]
                best_valid_metrics['loss'] = valid_loop_reports["loss"]

                tools = dict()
                tools['net'] = net
                tools['optimizer'] = optimizer
                tools['scheduler'] = scheduler

                accelerator.wait_for_everyone()

                # Set the path to save the model checkpoint.
                model_path = os.path.join(checkpoint_path, f'best_valid.pth')
                save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler,
                                configs=configs)
                if accelerator.is_main_process:
                    logging.info(f'\tsaving the best models in {model_path}')

        if epoch % configs.visualization_settings.do_every == 0:
            if accelerator.is_main_process:
                logging.info(f'\tstart visualization at epoch {epoch}')

            accelerator.wait_for_everyone()
            # Visualize the embeddings using T-SNE
            compute_visualization(net, visualization_loader, result_path, configs, logging, accelerator, epoch,
                                  optimizer)

    logging.info("Training is completed!\n")

    # log best valid gdtts
    if accelerator.is_main_process:
        logging.info(f"best valid gdtts: {best_valid_metrics['gdtts']:.4f}")
        logging.info(f"best valid mae: {best_valid_metrics['mae']:.4f}")
        logging.info(f"best valid rmse: {best_valid_metrics['rmse']:.4f}")
        logging.info(f"best valid loss: {best_valid_metrics['loss']:.4f}")

    train_writer.close()
    valid_writer.close()

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-VAE models.")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config_vqvae.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
