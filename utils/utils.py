import os
import glob
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
from accelerate.logging import get_logger
import sys
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import yaml
import h5py
from io import StringIO
from hydra import compose, initialize


def load_all_configs():
    # Initialize Hydra
    config_path = "configs"  # Path to your config directory

    with initialize(version_base=None, config_path="."):
        # Compose the configuration
        cfg = compose(
            config_name="config_vqvae",  # Main config
            overrides=[
                "+encoder=config_gcpnet_encoder",  # Load encoder config
                "+decoder=config_geometric_decoder",  # Load decoder config
            ]
        )

    return cfg


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


def get_logging(result_path, configs):
    # logger = log.getLogger(result_path)
    # logger.setLevel(log.INFO)

    logger = get_logger(__name__, log_level="INFO")

    log_file_path = os.path.join(result_path, "logs.txt")

    # Create a file handler (logs will be saved to 'training.log')
    file_handler = log.FileHandler(log_file_path, mode="w")
    file_handler.setLevel(log.INFO)

    # Define a log message format
    formatter = log.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Attach the file handler to the underlying logger
    logger.logger.addHandler(file_handler)

    # Stream handler (prints logs to the console)
    stream_handler = log.StreamHandler(sys.stdout)
    stream_handler.setLevel(log.INFO)
    stream_formatter = log.Formatter("%(asctime)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.logger.addHandler(stream_handler)

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
    optimizer_name = kwargs['configs'].optimizer.name
    optimizer = kwargs['optimizer']
    if optimizer_name == 'schedulerfree':
        optimizer.eval()

    # Save the models checkpoint.
    torch.save({
        'epoch': epoch,
        'model_state_dict': accelerator.unwrap_model(kwargs['net']).state_dict(),
        'optimizer_state_dict': kwargs['optimizer'].state_dict(),
        'scheduler_state_dict': kwargs['scheduler'].state_dict() if optimizer_name != 'schedulerfree' else "schedulerfree",
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

    # If the 'enabled' flag is True, load the saved models checkpoints.
    if configs.resume.enabled:
        model_checkpoint = torch.load(configs.resume.resume_path, map_location='cpu', weights_only=False)
        pretrained_state_dict = model_checkpoint['model_state_dict']
        pretrained_state_dict = {k.replace('_orig_mod.', ''): v for k, v in pretrained_state_dict.items()}

        # remove "vqvae.pos_embed_encoder" from the dict
        if 'vqvae.pos_embed_encoder' in pretrained_state_dict:
            # Handle AttributeError: 'VQVAETransformer' object has no attribute 'pos_embed_encoder'
            if hasattr(net.vqvae, 'pos_embed_encoder'):
                # Check if the vqvae.pos_embed_encoder of the checkpoint has not have the same shape as the model before removing
                if pretrained_state_dict['vqvae.pos_embed_encoder'].shape != net.vqvae.pos_embed_encoder.shape:
                    logging.info(f"Removing vqvae.pos_embed_encoder from the checkpoint as it has a different shape.")
                    pretrained_state_dict.pop('vqvae.pos_embed_encoder')
        loading_log = net.load_state_dict(pretrained_state_dict, strict=False)

        logging.info(f'Loading checkpoint log: {loading_log}')

        # If the saved checkpoint contains the optimizer and scheduler states and the epoch number,
        # resume training from the last saved epoch.
        if 'optimizer_state_dict' in model_checkpoint and 'scheduler_state_dict' in model_checkpoint and 'epoch' in model_checkpoint:
            if not configs.resume.restart_optimizer:
                optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
                logging.info('Optimizer is loaded to resume training!')

                # scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
                # if accelerator.main_process:
                #     logging.info('Scheduler is loaded to resume training!')

            # start_epoch = model_checkpoint['epoch'] + 1
            start_epoch = 1
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
    if scheduler is None and configs.optimizer.name.lower() != 'schedulerfree':
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
    else:
        logging.info('Using scheduler free optimizer')

    return optimizer, scheduler


def load_opt(model, config, logging):
    scheduler = None
    if config.optimizer.name.lower() == 'adabelief':
        opt = optim.AdaBelief(model.parameters(), lr=config.optimizer.lr, eps=config.optimizer.eps,
                              decoupled_decay=True,
                              weight_decay=config.optimizer.weight_decay, rectify=False)
    elif config.optimizer.name.lower() == 'adam' and config.optimizer.weight_decouple:
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
    elif config.optimizer.name.lower() == 'adam':
        opt = optim.Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay,
                         eps=config.optimizer.eps, betas=(config.optimizer.beta_1, config.optimizer.beta_2))

    elif config.optimizer.name.lower() == 'sgd' and config.optimizer.weight_decouple:
        raise ValueError('SGD with weight decouple is not supported yet')

    elif config.optimizer.name.lower() == 'sgd':
        opt = optim.SGD(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    elif config.optimizer.name.lower() == 'schedulerfree':
        import schedulefree
        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=float(config.optimizer.lr),
                                             warmup_steps=config.optimizer.decay.warmup,
                                             betas=(config.optimizer.beta_1, config.optimizer.beta_2),
                                             weight_decay=float(config.optimizer.weight_decay),
                                             eps=float(config.optimizer.eps))
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
    Path(os.path.join(result_path, 'pdb_files')).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file to the result directory.
    shutil.copy(config_file_path, result_path)

    # Return the path to the result directory.
    return result_path, checkpoint_path


def load_encoder_decoder_configs(configs, result_path):
    if configs.model.encoder.name == 'gvp_transformer':
        encoder_config_file_path = os.path.join('configs', 'config_gvp_transformer_encoder.yaml')
    elif configs.model.encoder.name == 'gcpnet':
        encoder_config_file_path = os.path.join('configs', 'config_gcpnet_encoder.yaml')
    else:
        raise ValueError('Unknown encoder')

    with open(encoder_config_file_path) as file:
        encoder_config_file = yaml.full_load(file)

    encoder_configs = Box(encoder_config_file)

    shutil.copy(encoder_config_file_path, result_path)

    if configs.model.decoder == 'geometric_decoder':
        decoder_config_file_path = os.path.join('configs', 'config_geometric_decoder.yaml')
    elif configs.model.decoder == 'gcpnet':
        decoder_config_file_path = os.path.join('configs', 'config_gcpnet_decoder.yaml')
    else:
        raise ValueError('Unknown decoder')

    with open(decoder_config_file_path) as file:
        decoder_config_file = yaml.full_load(file)

    decoder_configs = Box(decoder_config_file)

    shutil.copy(decoder_config_file_path, result_path)

    return encoder_configs, decoder_configs


def load_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        seq = f['seq'][()]
        n_ca_c_o_coord = f['N_CA_C_O_coord'][:]
        plddt_scores = f['plddt_scores'][:]
    return seq, n_ca_c_o_coord, plddt_scores


def save_backbone_pdb(coords, masks, save_path_prefix, atom_names=["N", "CA", "C"]):
    """
    Convert backbone (N, CA, C) atom coordinates to PDB files, one per sample in the batch.

    :param coords: (torch.Tensor[batch_size, n_residues, 3, 3]) Backbone atom coordinates (N, CA, C).
    :param masks: (torch.Tensor[batch_size, n_residues]) Masks corresponding to each residue.
    :param save_path_prefix: (str) Prefix for the path to save the PDB files.
                                  A suffix like '_sample_0.pdb', '_sample_1.pdb' will be added.
    :param atom_names: (list) List of atom names, e.g., ["N", "CA", "C"].
    """
    if coords.dim() == 3: # If a single sample is passed [n_residues, 3, 3]
        coords = coords.unsqueeze(0) # Add batch dimension
        masks = masks.unsqueeze(0) # Add batch dimension

    for i in range(coords.shape[0]):  # Loop over batch
        # Ensure save_path_prefix does not end with .pdb if we are adding _sample_idx.pdb
        current_save_path = f"{save_path_prefix}_sample_{i}.pdb"
        if save_path_prefix.lower().endswith(".pdb"):
             base_name = save_path_prefix[:-4] # Remove .pdb
             current_save_path = f"{base_name}_sample_{i}.pdb"


        with open(current_save_path, 'w') as pdb_file:
            atom_serial_number = 1
            for res_idx in range(coords.shape[1]):  # Loop over residues
                if masks[i, res_idx].item() == 1:  # Only write residues that are not masked
                    for atom_k in range(coords.shape[2]):  # Loop over N, CA, C atoms
                        atom_name_str = atom_names[atom_k]
                        x, y, z = coords[i, res_idx, atom_k].tolist()

                        # Basic PDB ATOM line format.
                        # You might need to adjust chain ID, residue name (UNK), etc., as needed.
                        # ATOM serial name altloc resname chain resseq icode X Y Z occupancy temp element
                        pdb_file.write(
                            f"ATOM  {atom_serial_number:5d} {atom_name_str:<4s} UNK A {res_idx + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name_str[0]:<2s}\n"
                        )
                        atom_serial_number += 1
            pdb_file.write("TER\n")
            pdb_file.write("END\n")


if __name__ == "__main__":

    bsz = 1
    num_res = 5
    pdb_path = "test.pdb"
    h5_path = "/DATA/renjz/data/swissprot_1024_h5"

    # Test ca_coords_to_pdb with random coordinates
    coordinates = torch.rand((bsz, num_res, 3))
    atom_masks = torch.zeros((bsz, num_res))
    atom_masks = torch.tensor([[1, 1, 0, 1, 1]])
    save_backbone_pdb(coordinates, atom_masks, pdb_path)
    # print(coordinates)

    # Test ca_coords_to_pdb with h5 structures
    n_samps = 1
    h5_samples = glob.glob(os.path.join(h5_path, '*.h5'))[:n_samps]
    batch_coords = []
    for i in range(n_samps):
        sample_path = h5_samples[i]
        sample = load_h5_file(sample_path)
        coords_list = sample[1].tolist()
        coords_tensor = torch.Tensor(coords_list)[:, 1, :]
        batch_coords.append(coords_tensor)
    coordinates = torch.stack(batch_coords)
    atom_masks = torch.ones((n_samps, coordinates.shape[1]))
    save_backbone_pdb(coordinates, atom_masks, pdb_path)
    print(coordinates)
