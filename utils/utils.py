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


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the models.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model, logging, description=""):
    """
    Prints the number of trainable parameters in the models.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"{description} trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


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
    return logger


def save_checkpoint(epoch: int, model_path: str, accelerator: Accelerator, **kwargs):
    """
    Save the models checkpoints during training.

    Args:
        epoch (int): The current epoch number.
        model_path (str): The path to save the models checkpoint.
        tools (dict): A dictionary containing the necessary tools for saving the models checkpoints.
        accelerator (Accelerator): Accelerator object.

    Returns:
        None
    """

    # Save the models checkpoint.
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
        net (nn.Module): The neural network models to load the saved checkpoints into.

    Returns:
        tuple: A tuple containing the loaded neural network models and the epoch to start training from.
    """
    start_epoch = 1

    # If the 'resume' flag is True, load the saved models checkpoints.
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
                        logging.info(
                            f'Copied overlapping parts of this layer: {name}, Checkpoint shape: {param.size()}, Model shape: {model_state_dict[name].size()}')
                else:
                    if accelerator.is_main_process:
                        logging.info(
                            f'Ignore {name} layer, missmatch: Checkpoint shape: {param.size()}, Model shape: {model_state_dict[name].size()}')

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
        # opt = eval('torch.optim.' + config.optimizer.name)(models.parameters(), lr=config.optimizer.lr, eps=eps,
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


def load_configs_gvp(config):
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

    # set configs value to default if doesn't have the attr
    if not hasattr(tree_config.model.struct_encoder, "use_seq"):
        tree_config.model.struct_encoder.use_seq = None
        tree_config.model.struct_encoder.use_seq.enable = False
        tree_config.model.struct_encoder.use_seq.seq_embed_mode = "embedding"
        tree_config.model.struct_encoder.use_seq.seq_embed_dim = 20

    if not hasattr(tree_config.model.struct_encoder, "top_k"):
        tree_config.model.struct_encoder.top_k = 30  # default

    if not hasattr(tree_config.model.struct_encoder, "gvp_num_layers"):
        tree_config.model.struct_encoder.gvp_num_layers = 3  # default

    if not hasattr(tree_config.model.struct_encoder,
                   "use_rotary_embeddings"):  # configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_rotary_embeddings = False

    if not hasattr(tree_config.model.struct_encoder, "rotary_mode"):
        tree_config.model.struct_encoder.rotary_mode = 1

    if not hasattr(tree_config.model.struct_encoder,
                   "use_foldseek"):  # configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek = False

    if not hasattr(tree_config.model.struct_encoder,
                   "use_foldseek_vector"):  # configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek_vector = False

    if not hasattr(tree_config.model.struct_encoder, "num_rbf"):
        tree_config.model.struct_encoder.num_rbf = 16  # default

    if not hasattr(tree_config.model.struct_encoder, "num_positional_embeddings"):
        tree_config.model.struct_encoder.num_positional_embeddings = 16  # default

    if not hasattr(tree_config.model.struct_encoder, "node_h_dim"):
        tree_config.model.struct_encoder.node_h_dim = (100, 32)  # default
    else:
        tree_config.model.struct_encoder.node_h_dim = ast.literal_eval(tree_config.model.struct_encoder.node_h_dim)

    if not hasattr(tree_config.model.struct_encoder, "edge_h_dim"):
        tree_config.model.struct_encoder.edge_h_dim = (32, 1)  # default
    else:
        tree_config.model.struct_encoder.edge_h_dim = ast.literal_eval(tree_config.model.struct_encoder.edge_h_dim)

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
