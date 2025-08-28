import argparse
import numpy as np
import yaml
import os
import torch
from utils.custom_losses import calculate_decoder_loss, compute_grad_norm
from utils.utils import (
    save_backbone_pdb,
    load_configs,
    load_checkpoints,
    prepare_saving_dir,
    get_logging,
    prepare_optimizer,
    prepare_tensorboard,
    save_checkpoint,
    load_encoder_decoder_configs)
from utils.metrics import GDTTS, TMScore
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from visualization.main import compute_visualization
from tqdm import tqdm
import time
import torchmetrics
from data.dataset import prepare_gcpnet_vqvae_dataloaders
from models.super_model import prepare_model


def train_loop(net, train_loader, epoch, **kwargs):
    accelerator = kwargs.pop('accelerator')
    optimizer = kwargs.pop('optimizer')
    scheduler = kwargs.pop('scheduler')
    configs = kwargs.pop('configs')
    optimizer_name = configs.optimizer.name
    writer = kwargs.pop('writer')
    logging = kwargs.pop('logging')
    profiler = kwargs.pop('profiler')
    profile_train_loop = kwargs.pop('profile_train_loop')
    alpha = configs.model.vqvae.vector_quantization.alpha
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    accum_iter = configs.train_settings.grad_accumulation
    alignment_strategy = configs.train_settings.losses.alignment_strategy

    # Prepare metrics for evaluation
    rmsd = torchmetrics.MeanSquaredError(squared=False)
    mae = torchmetrics.MeanAbsoluteError()
    gdtts = GDTTS()
    tm_score_metric = TMScore()

    rmsd.to(accelerator.device)
    mae.to(accelerator.device)
    gdtts.to(accelerator.device)
    tm_score_metric.to(accelerator.device)

    optimizer.zero_grad()

    train_total_loss = 0.0
    train_rec_loss = 0.0
    train_cmt_loss = 0.0
    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0
    epoch_unique_indices_collector = set()  # Added for collecting unique indices
    counter = 0
    global_step = kwargs.get('global_step', 0)

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / accum_iter))),
                        leave=False, disable=not (configs.tqdm_progress_bar and accelerator.is_main_process))
    progress_bar.set_description(f"Epoch {epoch}")

    net.train()
    if optimizer_name == 'schedulerfree':
        optimizer.train()
    for i, data in enumerate(train_loader):
        with accelerator.accumulate(net):
            if profile_train_loop:
                profiler.step()
                if i >= 1 + 1 + 30:
                    logging.info("Profiler finished, exiting train step loop.")
                    break

            labels = data['target_coords']
            masks = torch.logical_and(data['masks'], data['nan_masks'])

            optimizer.zero_grad()
            net_outputs, indices, commit_loss = net(data)

            outputs, dir_loss_logits, dist_loss_logits = net_outputs

            # Compute the loss components
            loss_dict, trans_pred_coords, trans_true_coords = calculate_decoder_loss(
                x_predicted=outputs.reshape(outputs.shape[0], outputs.shape[1], 3, 3),
                x_true=labels.reshape(labels.shape[0], labels.shape[1], 3, 3),
                masks=masks.float(),
                configs=configs,
                seq=data["inverse_folding_labels"],
                dir_loss_logits=dir_loss_logits,
                dist_loss_logits=dist_loss_logits,
                alignment_strategy=alignment_strategy
            )
            rec_loss = loss_dict['rec_loss']

            loss = rec_loss + alpha * commit_loss

            # Log per-loss gradient norms before backward
            if accelerator.sync_gradients and accelerator.is_main_process and global_step % configs.train_settings.gradient_norm_logging_freq == 0 and configs.tensorboard_log:
                # reconstruction components
                if configs.train_settings.losses.mse.enabled:
                    gn = compute_grad_norm(loss_dict['mse_loss'], net.parameters())
                    writer.add_scalar('gradient norm/mse', gn.item(), global_step)
                if configs.train_settings.losses.backbone_distance.enabled:
                    gn = compute_grad_norm(loss_dict['backbone_distance_loss'], net.parameters())
                    writer.add_scalar('gradient norm/backbone_distance', gn.item(), global_step)
                if configs.train_settings.losses.backbone_direction.enabled:
                    gn = compute_grad_norm(loss_dict['backbone_direction_loss'], net.parameters())
                    writer.add_scalar('gradient norm/backbone_direction', gn.item(), global_step)
                if configs.train_settings.losses.binned_direction_classification.enabled:
                    gn = compute_grad_norm(loss_dict['binned_direction_classification_loss'], net.parameters())
                    writer.add_scalar('gradient norm/binned_direction_classification', gn.item(), global_step)
                if configs.train_settings.losses.binned_distance_classification.enabled:
                    gn = compute_grad_norm(loss_dict['binned_distance_classification_loss'], net.parameters())
                    writer.add_scalar('gradient norm/binned_distance_classification', gn.item(), global_step)

                if configs.model.vqvae.vector_quantization.enabled:
                    # commitment loss
                    gn = compute_grad_norm(alpha * commit_loss, net.parameters())
                    writer.add_scalar('gradient norm/commit', gn.item(), global_step)

            if accelerator.is_main_process and epoch % configs.train_settings.save_pdb_every == 0 and epoch != 0 and i == 0:
                logging.info(f"Building PDB files for training data in epoch {epoch}")
                save_backbone_pdb(trans_pred_coords.detach(), masks, data['pid'],
                                  os.path.join(kwargs['result_path'], 'pdb_files',
                                               f'train_outputs_epoch_{epoch}_step_{i + 1}'))
                save_backbone_pdb(trans_true_coords.detach().squeeze(), masks, data['pid'],
                                  os.path.join(kwargs['result_path'], 'pdb_files', f'train_labels_step_{i + 1}'))
                logging.info("PDB files are built")

            # Compute the loss
            masked_outputs = trans_pred_coords[masks]
            masked_labels = trans_true_coords[masks]

            # Denormalize the outputs and labels
            masked_outputs = (masked_outputs).reshape(-1, 3)
            masked_labels = (masked_labels).reshape(-1, 3)

            if masked_outputs.numel() > 0:
                # --->>> UPDATE WITH LOCAL TENSORS <<<---
                # Pass the tensors directly from the current GPU.
                # torchmetrics + accelerate handle the sync later.
                mae.update(masked_outputs.detach(), masked_labels.detach())
                rmsd.update(masked_outputs.detach(), masked_labels.detach())
                gdtts.update(masked_outputs.detach(), masked_labels.detach())
                # Extract C-alpha coordinates for TMScore
                pred_ca_coords = trans_pred_coords[:, :, 1, :].detach()
                true_ca_coords = trans_true_coords[:, :, 1, :].detach()
                tm_score_metric.update(pred_ca_coords, true_ca_coords, masks.detach().bool())

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_rec_loss = accelerator.gather(rec_loss.repeat(configs.train_settings.batch_size)).mean()
            train_rec_loss += avg_rec_loss.item() / accum_iter

            avg_cmt_loss = accelerator.gather(commit_loss.repeat(configs.train_settings.batch_size)).mean()
            train_cmt_loss += avg_cmt_loss.item() / accum_iter

            train_total_loss = train_rec_loss + alpha * train_cmt_loss

            gathered_indices = accelerator.gather(indices)
            epoch_unique_indices_collector.update(gathered_indices.unique().cpu().tolist())

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if global_step % configs.train_settings.gradient_norm_logging_freq == 0:
                    # Calculate the gradient norm every configs.train_settings.gradient_norm_logging_freq steps
                    grad_norm = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2) for p in net.parameters() if p.grad is not None]),
                        2)
                    if accelerator.is_main_process and configs.tensorboard_log:
                        writer.add_scalar('gradient norm/total', grad_norm.item(), global_step)

                if optimizer_name != 'schedulerfree':
                    accelerator.clip_grad_norm_(net.parameters(), configs.optimizer.grad_clip_norm)

            if optimizer_name != 'schedulerfree':
                optimizer.step()
                scheduler.step()
            else:
                optimizer.step()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            counter += 1

            # Keep track of total combined loss, total reconstruction loss, and total commit loss
            total_loss += train_total_loss
            total_rec_loss += train_rec_loss
            total_cmt_loss += train_cmt_loss
            # epoch_unique_indices_collector.update(indices.unique().cpu().tolist()) # Removed from here

            train_total_loss = 0.0
            train_rec_loss = 0.0
            train_cmt_loss = 0.0

            if accelerator.is_main_process and configs.tensorboard_log:
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

            progress_bar.set_description(f"epoch {epoch} "
                                         + f"[loss: {total_loss / counter:.3f}, "
                                         + f"rec loss: {total_rec_loss / counter:.3f}, "
                                         + f"cmt loss: {total_cmt_loss / counter:.3f}]")

        progress_bar.set_postfix(
            {
                "lr": optimizer.param_groups[0]['lr'],
                "step_loss": loss.detach().item(),
                "rec_loss": rec_loss.detach().item(),
                "cmt_loss": commit_loss.detach().item(),
                "global_step": global_step
            }
        )

    # Compute average losses and metrics
    avg_loss = total_loss / counter
    avg_rec_loss = total_rec_loss / counter
    denormalized_rec_mae = mae.compute().cpu().item()
    denormalized_rec_rmsd = rmsd.compute().cpu().item()
    gdtts_score = gdtts.compute().cpu().item()
    tm_score = tm_score_metric.compute().cpu().item()
    avg_cmt_loss = total_cmt_loss / counter

    # Calculate global unique codebook activation
    num_truly_unique_codes = len(epoch_unique_indices_collector)
    if codebook_size > 0:
        avg_activation = num_truly_unique_codes / codebook_size
    else:
        avg_activation = 0.0  # Avoid division by zero

    # Log the metrics to TensorBoard
    if accelerator.is_main_process and configs.tensorboard_log:
        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('rec_loss', avg_rec_loss, epoch)
        writer.add_scalar('mae', denormalized_rec_mae, epoch)
        writer.add_scalar('rmsd', denormalized_rec_rmsd, epoch)
        writer.add_scalar('gdtts', gdtts_score, epoch)
        writer.add_scalar('tm_score', tm_score, epoch)
        writer.add_scalar('cmt_loss', avg_cmt_loss, epoch)
        writer.add_scalar('codebook_activation', np.round(avg_activation * 100, 1), epoch)

    # Reset the metrics for the next epoch
    mae.reset()
    rmsd.reset()
    gdtts.reset()
    tm_score_metric.reset()

    return_dict = {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "mae": denormalized_rec_mae,
        "rmsd": denormalized_rec_rmsd,
        "gdtts": gdtts_score,
        "tm_score": tm_score,
        "cmt_loss": avg_cmt_loss,
        "activation": np.round(avg_activation * 100, 1),
        "counter": counter,
        "global_step": global_step
    }
    return return_dict


def valid_loop(net, valid_loader, epoch, **kwargs):
    optimizer = kwargs.pop('optimizer')
    configs = kwargs.pop('configs')
    optimizer_name = configs.optimizer.name
    accelerator = kwargs.pop('accelerator')
    writer = kwargs.pop('writer')
    logging = kwargs.pop('logging')
    alpha = configs.model.vqvae.vector_quantization.alpha
    codebook_size = configs.model.vqvae.vector_quantization.codebook_size
    alignment_strategy = configs.train_settings.losses.alignment_strategy

    # Prepare metrics for evaluation, initialized once for the validation epoch
    mae_metric_val = torchmetrics.MeanAbsoluteError().to(accelerator.device)
    rmsd_metric_val = torchmetrics.MeanSquaredError(squared=False).to(accelerator.device)
    gdtts_metric_val = GDTTS().to(accelerator.device)
    tm_score_metric_val = TMScore().to(accelerator.device)  # Ensure TMScore is initialized

    total_loss = 0.0
    total_rec_loss = 0.0
    total_cmt_loss = 0.0
    epoch_unique_indices_collector = set()
    counter = 0

    optimizer.zero_grad()

    # Initialize the progress bar using tqdm
    progress_bar = tqdm(range(0, int(len(valid_loader))),
                        leave=False, disable=not (configs.tqdm_progress_bar and accelerator.is_main_process))
    progress_bar.set_description(f"Validation epoch {epoch}")

    net.eval()
    if optimizer_name != 'schedulerfree':
        optimizer.eval()
    for i, data in enumerate(valid_loader):
        with torch.inference_mode():
            labels = data['target_coords']
            masks = torch.logical_and(data['masks'], data['nan_masks'])

            optimizer.zero_grad()
            net_outputs, indices, commit_loss = net(data)

            gathered_indices = accelerator.gather(indices)
            epoch_unique_indices_collector.update(gathered_indices.unique().cpu().tolist())

            outputs, dir_loss_logits, dist_loss_logits = net_outputs

            # Compute the loss components
            loss_dict, trans_pred_coords, trans_true_coords = calculate_decoder_loss(
                x_predicted=outputs.reshape(outputs.shape[0], outputs.shape[1], 3, 3),
                x_true=labels.reshape(labels.shape[0], labels.shape[1], 3, 3),
                masks=masks.float(),
                configs=configs,
                seq=data["inverse_folding_labels"],
                dir_loss_logits=dir_loss_logits,
                dist_loss_logits=dist_loss_logits,
                alignment_strategy=alignment_strategy
            )
            rec_loss = loss_dict['rec_loss']
            loss = rec_loss + alpha * commit_loss

            if accelerator.is_main_process and epoch % configs.valid_settings.save_pdb_every == 0 and epoch != 0 and i == 0:
                logging.info(f"Building PDB files for validation data in epoch {epoch}")
                save_backbone_pdb(trans_pred_coords.detach(), masks, data['pid'],
                                  os.path.join(kwargs['result_path'], 'pdb_files',
                                               f'valid_outputs_epoch_{epoch}_step_{i + 1}'))
                save_backbone_pdb(trans_true_coords.detach(), masks, data['pid'],
                                  os.path.join(kwargs['result_path'], 'pdb_files', f'valid_labels_step_{i + 1}'))
                logging.info("PDB files are built")

            # Extract masked coordinates
            masked_outputs = trans_pred_coords[masks].reshape(-1, 3)
            masked_labels = trans_true_coords[masks].reshape(-1, 3)

            # Compute local metrics
            # Using torchmetrics for MAE and RMSD
            if masked_outputs.numel() > 0:
                mae_metric_val.update(masked_outputs.detach(), masked_labels.detach())
                rmsd_metric_val.update(masked_outputs.detach(), masked_labels.detach())

            # For GDTTS (and TM-score if used), use C-alpha coordinates
            pred_ca_coords_val = trans_pred_coords[:, :, 1, :].detach()  # Shape: (B, L, 3)
            true_ca_coords_val = trans_true_coords[:, :, 1, :].detach()  # Shape: (B, L, 3)
            current_masks_val = masks.detach()  # Shape: (B, L)

            if pred_ca_coords_val.numel() > 0 and true_ca_coords_val.numel() > 0:
                # Apply masks to C-alpha coordinates before GDTTS update
                masked_pred_ca_val = pred_ca_coords_val[current_masks_val]  # Shape: (M, 3)
                masked_true_ca_val = true_ca_coords_val[current_masks_val]  # Shape: (M, 3)

                if masked_pred_ca_val.numel() > 0:  # Ensure there are residues after masking
                    gdtts_metric_val.update(masked_pred_ca_val, masked_true_ca_val)

                # Update TMScore with C-alpha coordinates and original masks
                # The TMScore class handles masking internally
                tm_score_metric_val.update(pred_ca_coords_val, true_ca_coords_val, current_masks_val)

        progress_bar.update(1)
        counter += 1

        batch_size = data['target_coords'].shape[0]
        total_loss += accelerator.gather(loss.repeat(batch_size)).mean().item()
        total_rec_loss += accelerator.gather(rec_loss.repeat(batch_size)).mean().item()
        total_cmt_loss += accelerator.gather(commit_loss.repeat(batch_size)).mean().item()

        progress_bar.set_description(f"validation epoch {epoch} "
                                     + f"[loss: {total_loss / counter:.3f}, "
                                     + f"rec loss: {total_rec_loss / counter:.3f}, "
                                     + f"cmt loss: {total_cmt_loss / counter:.3f}]")

        # Compute average losses
    avg_loss = total_loss / counter
    avg_rec_loss = total_rec_loss / counter
    avg_cmt_loss = total_cmt_loss / counter

    # Calculate global unique codebook activation
    num_truly_unique_codes = len(epoch_unique_indices_collector)
    if codebook_size > 0:
        avg_activation = num_truly_unique_codes / codebook_size
    else:
        avg_activation = 0.0  # Avoid division by zero

    # Compute final metrics using torchmetrics objects
    denormalized_rec_mae = mae_metric_val.compute().cpu().item()
    denormalized_rec_rmsd = rmsd_metric_val.compute().cpu().item()
    gdtts_score = gdtts_metric_val.compute().cpu().item()
    tm_score_val = tm_score_metric_val.compute().cpu().item()  # Compute TM-score

    # Reset metrics for the next epoch
    mae_metric_val.reset()
    rmsd_metric_val.reset()
    gdtts_metric_val.reset()
    tm_score_metric_val.reset()  # Reset TM-score metric

    # Log metrics to TensorBoard
    if accelerator.is_main_process and configs.tensorboard_log:
        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('rec_loss', avg_rec_loss, epoch)
        writer.add_scalar('mae', denormalized_rec_mae, epoch)
        writer.add_scalar('rmsd', denormalized_rec_rmsd, epoch)
        writer.add_scalar('gdtts', gdtts_score, epoch)
        writer.add_scalar('tm_score', tm_score_val, epoch)  # Log TM-score
        writer.add_scalar('codebook_activation', np.round(avg_activation * 100, 1), epoch)

    return_dict = {
        "loss": avg_loss,
        "rec_loss": avg_rec_loss,
        "mae": denormalized_rec_mae,
        "rmsd": denormalized_rec_rmsd,
        "gdtts": gdtts_score,
        "tm_score": tm_score_val,  # Add TM-score to return dict
        "activation": np.round(avg_activation * 100, 1),
        "counter": counter,
    }
    return return_dict


def main(dict_config, config_file_path):
    configs = load_configs(dict_config)
    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    # Set find_unused_parameters to True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=configs.find_unused_parameters)
    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=configs.dispatch_batches,
        even_batches=configs.even_batches,
        non_blocking=configs.non_blocking,
        split_batches=configs.split_batches,
        # use_stateful_dataloader=True
    )
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dataloader_config=dataloader_config
    )

    # Initialize paths to avoid unassigned variable warnings
    result_path, checkpoint_path = None, None

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)
        paths = [result_path, checkpoint_path]
    else:
        # Initialize with placeholders.
        paths = [None, None]

    if accelerator.num_processes > 1:
        import torch.distributed as dist
        # Broadcast the list of strings from the main process (src=0) to all others.
        dist.broadcast_object_list(paths, src=0)

        # Now every process has the shared values.
        result_path, checkpoint_path = paths

    encoder_configs, decoder_configs = load_encoder_decoder_configs(configs, result_path)

    logging = get_logging(result_path, configs)

    train_dataloader, valid_dataloader, visualization_loader = prepare_gcpnet_vqvae_dataloaders(
        logging, accelerator, configs, encoder_configs=encoder_configs, decoder_configs=decoder_configs
    )
    logging.info('preparing dataloaders are done')

    net = prepare_model(
        configs, logging,
        encoder_configs=encoder_configs,
        decoder_configs=decoder_configs
    )
    logging.info('preparing models is done')

    optimizer, scheduler = prepare_optimizer(net, configs, len(train_dataloader), logging)
    logging.info('preparing optimizer is done')

    net, start_epoch = load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator)

    # compile models to train faster and efficiently
    if configs.model.compile_model:
        if hasattr(net, 'vqvae'):
            net.vqvae = torch.compile(net.vqvae)
            logging.info('VQVAE component compiled.')

    net, optimizer, train_dataloader, valid_dataloader, visualization_loader, scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, valid_dataloader, visualization_loader, scheduler
    )

    net.to(accelerator.device)

    if accelerator.is_main_process:
        # initialize tensorboards
        train_writer, valid_writer = prepare_tensorboard(result_path)
    else:
        train_writer, valid_writer = None, None

    if accelerator.is_main_process:
        train_steps = np.ceil(len(train_dataloader) / configs.train_settings.grad_accumulation)
        logging.info(f'number of train steps per epoch: {int(train_steps)}')

    # Maybe monitor resource usage during training.
    prof = None
    profile_train_loop = configs.train_settings.profile_train_loop

    if profile_train_loop:
        from pathlib import Path
        train_profile_path = os.path.join(result_path, 'train', 'profile')
        Path(train_profile_path).mkdir(parents=True, exist_ok=True)
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=30, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(train_profile_path),
            profile_memory=True,
        )
        prof.start()

    # Use this to keep track of the global step across all processes.
    # This is useful for continuing training from a checkpoint.
    global_step = 0
    best_valid_metrics = {'gdtts': 0.0, 'mae': 1000.0, 'rmsd': 1000.0, 'lddt': 0.0, 'loss': 1000.0}
    for epoch in range(1, configs.train_settings.num_epochs + 1):
        start_time = time.time()
        training_loop_reports = train_loop(net, train_dataloader, epoch,
                                           accelerator=accelerator,
                                           optimizer=optimizer,
                                           scheduler=scheduler, configs=configs,
                                           logging=logging, global_step=global_step,
                                           writer=train_writer, result_path=result_path,
                                           profiler=prof, profile_train_loop=profile_train_loop)

        if profile_train_loop:
            prof.stop()
            logging.info("Profiler stopped, exiting train epoch loop.")
            break

        end_time = time.time()
        training_time = end_time - start_time
        logging.info(
            f'epoch {epoch} ({training_loop_reports["counter"]} steps) - time {np.round(training_time, 2)}s, '
            f'global steps {training_loop_reports["global_step"]}, loss {training_loop_reports["loss"]:.4f}, '
            f'rec loss {training_loop_reports["rec_loss"]:.4f}, '
            f'mae {training_loop_reports["mae"]:.4f}, '
            f'rmsd {training_loop_reports["rmsd"]:.4f}, '
            f'gdtts {training_loop_reports["gdtts"]:.4f}, '
            f'tm_score {training_loop_reports["tm_score"]:.4f}, '
            f'cmt loss {training_loop_reports["cmt_loss"]:.4f}, '
            f'activation {training_loop_reports["activation"]:.1f}')

        global_step = training_loop_reports["global_step"]
        accelerator.wait_for_everyone()

        if epoch % configs.checkpoints_every == 0:
            tools = dict()
            tools['net'] = net
            tools['optimizer'] = optimizer
            tools['scheduler'] = scheduler

            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                # Set the path to save the models checkpoint.
                model_path = os.path.join(checkpoint_path, f'epoch_{epoch}.pth')
                save_checkpoint(epoch, model_path, accelerator, net=net, optimizer=optimizer, scheduler=scheduler,
                                configs=configs)
                logging.info(f'\tcheckpoint models in {model_path}')

        if epoch % configs.valid_settings.do_every == 0:
            start_time = time.time()
            valid_loop_reports = valid_loop(net, valid_dataloader, epoch,
                                            accelerator=accelerator,
                                            optimizer=optimizer,
                                            scheduler=scheduler, configs=configs,
                                            logging=logging, global_step=global_step,
                                            writer=valid_writer, result_path=result_path)
            end_time = time.time()
            valid_time = end_time - start_time
            accelerator.wait_for_everyone()
            logging.info(
                f'validation epoch {epoch} ({valid_loop_reports["counter"]} steps) - time {np.round(valid_time, 2)}s, '
                f'loss {valid_loop_reports["loss"]:.4f}, '
                f'rec loss {valid_loop_reports["rec_loss"]:.4f}, '
                f'mae {valid_loop_reports["mae"]:.4f}, '
                f'rmsd {valid_loop_reports["rmsd"]:.4f}, '
                f'gdtts {valid_loop_reports["gdtts"]:.4f}, '
                f'tm_score {valid_loop_reports["tm_score"]:.4f}, '
                f'activation {valid_loop_reports["activation"]:.1f}'
                # f'lddt {valid_loop_reports["lddt"]:.4f}'
            )

            # Check valid metric to save the best model
            if valid_loop_reports["rmsd"] < best_valid_metrics['rmsd']:
                best_valid_metrics['gdtts'] = valid_loop_reports["gdtts"]
                best_valid_metrics['mae'] = valid_loop_reports["mae"]
                best_valid_metrics['rmsd'] = valid_loop_reports["rmsd"]
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
                logging.info(f'\tsaving the best models in {model_path}')
                logging.info(f'\tbest valid rmsd: {best_valid_metrics["rmsd"]:.4f}')

        if epoch % configs.visualization_settings.do_every == 0:
            logging.info(f'\tstart visualization at epoch {epoch}')

            accelerator.wait_for_everyone()
            # Visualize the embeddings using T-SNE
            logging.info(f'Calling compute_visualization for epoch {epoch}')
            compute_visualization(net, visualization_loader, result_path, configs, logging, accelerator, epoch,
                                  optimizer)

    logging.info("Training is completed!\n")

    # log best valid gdtts
    logging.info(f"best valid gdtts: {best_valid_metrics['gdtts']:.4f}")
    logging.info(f"best valid mae: {best_valid_metrics['mae']:.4f}")
    logging.info(f"best valid rmsd: {best_valid_metrics['rmsd']:.4f}")
    logging.info(f"best valid loss: {best_valid_metrics['loss']:.4f}")

    if accelerator.is_main_process:
        train_writer.close()
        valid_writer.close()

    accelerator.wait_for_everyone()
    accelerator.free_memory()
    accelerator.end_training()
    torch.cuda.empty_cache()
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VQ-VAE models.")
    parser.add_argument("--config_path", "-c", help="The location of config file",
                        default='./configs/config_vqvae.yaml')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(config_file, config_path)
