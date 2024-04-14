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
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


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
