import os
import datetime
import shutil
import ast
import logging as log
from pathlib import Path
from box import Box
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import numpy as np
from accelerate import Accelerator
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import math
import h5py
from io import StringIO


def get_logging(result_path):
    logger = log.getLogger(result_path)
    logger.setLevel(log.INFO)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger


def get_dummy_logger():
    # Create a logger object
    logger = log.getLogger('dummy')
    logger.setLevel(log.INFO)

    # Create a string buffer to hold the logs
    log_buffer = StringIO()

    # Create a stream handler that writes to the string buffer
    handler = log.StreamHandler(log_buffer)
    formatter = log.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optionally disable propagation to prevent logging on the parent logger
    logger.propagate = False

    # Return both logger and buffer so you can inspect logs as needed
    return logger, log_buffer


def save_checkpoint(epoch: int, model_path: str, accelerator: Accelerator, **kwargs):
    """
    Save the model checkpoints during training.

    Args:
        epoch (int): The current epoch number.
        model_path (str): The path to save the model checkpoint.
        tools (dict): A dictionary containing the necessary tools for saving the model checkpoints.
        accelerator (Accelerator): Accelerator object.

    Returns:
        None
    """

    # Save the model checkpoint.
    torch.save({
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(kwargs['net']).state_dict(),
        'optimizer_state_dict': accelerator.unwrap_model(kwargs['optimizer'].state_dict()),
        'scheduler_state_dict': accelerator.unwrap_model(kwargs['scheduler'].state_dict()),
    }, model_path)


def load_checkpoints_simple(checkpoint_path, net):
    model_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrained_state_dict = model_checkpoint['model_state_dict']
    net.load_state_dict(pretrained_state_dict, strict=True)
    return net


def load_checkpoints(configs, optimizer, scheduler, logging, net, accelerator):
    """
    Load saved checkpoints from a previous training session.

    Args:
        configs: A python box object containing the configuration options.
        optimizer (Optimizer): The optimizer to resume training with.
        scheduler (Scheduler): The learning rate scheduler to resume training with.
        logging (Logger): The logger to use for logging messages.
        net (nn.Module): The neural network model to load the saved checkpoints into.

    Returns:
        tuple: A tuple containing the loaded neural network model and the epoch to start training from.
    """
    start_epoch = 1

    # If the 'resume' flag is True, load the saved model checkpoints.
    if configs.resume.resume:
        model_checkpoint = torch.load(configs.resume.resume_path, map_location='cpu')
        # state_dict = model_checkpoint['model_state_dict']
        # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        pretrained_state_dict = model_checkpoint['model_state_dict']
        model_state_dict = net.state_dict()

        if configs.resume.handle_shape_missmatch:
            if accelerator.is_main_process:
                logging.info(f'Consider handling shape miss match to reload the checkpoint.')

        for name, param in pretrained_state_dict.items():
            if name in model_state_dict:
                if model_state_dict[name].size() == param.size():
                    model_state_dict[name].copy_(param)
                elif configs.resume.handle_shape_missmatch:
                    # Copy only the overlapping parts of the tensor
                    # Assumes the mismatch is in the first dimension
                    if len(model_state_dict[name].size()) == 2:
                        min_size = min(model_state_dict[name].size(0), param.size(0))
                        model_state_dict[name][:min_size].copy_(param[:min_size])
                    else:
                        min_size = min(model_state_dict[name].size(1), param.size(1))
                        model_state_dict[name][:, :min_size].copy_(param[:, :min_size])
                    if accelerator.is_main_process:
                        logging.info(f'Copied overlapping parts of this layer: {name}, Checkpoint shape: {param.size()}, Model shape: {model_state_dict[name].size()}')
                else:
                    if accelerator.is_main_process:
                        logging.info(f'Ignore {name} layer, missmatch: Checkpoint shape: {param.size()}, Model shape: {model_state_dict[name].size()}')

        loading_log = net.load_state_dict(model_state_dict, strict=False)
        if accelerator.is_main_process:
            logging.info(f'Loading checkpoint log: {loading_log}')

        # If the saved checkpoint contains the optimizer and scheduler states and the epoch number,
        # resume training from the last saved epoch.
        if 'optimizer_state_dict' in model_checkpoint and 'scheduler_state_dict' in model_checkpoint and 'epoch' in model_checkpoint:
            if not configs.resume.restart_optimizer:
                optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                if accelerator.is_main_process:
                    logging.info('Optimizer is loaded to resume training!')

                # scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
                # if accelerator.main_process:
                #     logging.info('Scheduler is loaded to resume training!')

            # start_epoch = model_checkpoint['epoch'] + 1
            start_epoch = 1
        if accelerator.is_main_process:
            logging.info('Model is loaded to resume training!')
    return net, start_epoch


def prepare_tensorboard(result_path):
    train_path = os.path.join(result_path, 'train')
    val_path = os.path.join(result_path, 'val')
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)

    train_log_path = os.path.join(train_path, 'tensorboard')
    train_writer = SummaryWriter(train_log_path)

    val_log_path = os.path.join(val_path, 'tensorboard')
    val_writer = SummaryWriter(val_log_path)

    return train_writer, val_writer


def prepare_optimizer(net, configs, num_train_samples, logging):
    optimizer, scheduler = load_opt(net, configs, logging)
    if scheduler is None:
        whole_steps = np.ceil(
            num_train_samples / configs.train_settings.grad_accumulation
        ) * configs.train_settings.num_epochs / configs.optimizer.decay.num_restarts
        first_cycle_steps = np.ceil(whole_steps / configs.optimizer.decay.num_restarts)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=1.0,
            max_lr=configs.optimizer.lr,
            min_lr=configs.optimizer.decay.min_lr,
            warmup_steps=configs.optimizer.decay.warmup,
            gamma=configs.optimizer.decay.gamma)

    return optimizer, scheduler


def load_opt(model, config, logging):
    scheduler = None
    if config.optimizer.name.lower() == 'adabelief':
        opt = optim.AdaBelief(model.parameters(), lr=config.optimizer.lr, eps=config.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=config.optimizer.weight_decay, rectify=False)
    elif config.optimizer.name.lower() == 'adam':
        # opt = eval('torch.optim.' + config.optimizer.name)(model.parameters(), lr=config.optimizer.lr, eps=eps,
        #                                       weight_decay=config.optimizer.weight_decay)
        if config.optimizer.use_8bit_adam:
            import bitsandbytes
            logging.info('use 8-bit adamw')
            opt = bitsandbytes.optim.AdamW8bit(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps),
            )
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=float(config.optimizer.lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                weight_decay=float(config.optimizer.weight_decay),
                eps=float(config.optimizer.eps)
            )

    else:
        raise ValueError('wrong optimizer')
    return opt, scheduler


def load_configs(config):
    """
        Load the configuration file and convert the necessary values to floats.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            The updated configuration dictionary with float values.
        """

    # Convert the dictionary to a Box object for easier access to the values.
    tree_config = Box(config)

    # Convert the necessary values to floats.
    tree_config.optimizer.lr = float(tree_config.optimizer.lr)
    tree_config.optimizer.decay.min_lr = float(tree_config.optimizer.decay.min_lr)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    
    return tree_config


def prepare_saving_dir(configs, config_file_path):
    """
    Prepare a directory for saving a training results.

    Args:
        configs: A python box object containing the configuration options.
        config_file_path: Directory of configuration file.

    Returns:
        str: The path to the directory where the results will be saved.
    """
    # Create a unique identifier for the run based on the current time.
    run_id = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    # Add '_evaluation' to the run_id if the 'evaluate' flag is True.
    # if configs.evaluate:
    #     run_id += '_evaluation'

    # Create the result directory and the checkpoint subdirectory.
    result_path = os.path.abspath(os.path.join(configs.result_path, run_id))
    checkpoint_path = os.path.join(result_path, 'checkpoints')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    # Return the path to the result directory.
    return result_path, checkpoint_path


def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        plddt_scores = f['plddt_scores'][:]
    return seq, n_ca_c_o_coord, plddt_scores


