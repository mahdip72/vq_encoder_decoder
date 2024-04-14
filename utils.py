import os
import datetime
import shutil
import ast
from pathlib import Path
from box import Box
from torch.utils.tensorboard import SummaryWriter


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
    
    #set configs value to default if doesn't have the attr
    if not hasattr(tree_config.model.struct_encoder, "use_seq"):
        tree_config.model.struct_encoder.use_seq = None
        tree_config.model.struct_encoder.use_seq.enable = False
        tree_config.model.struct_encoder.use_seq.seq_embed_mode = "embedding"
        tree_config.model.struct_encoder.use_seq.seq_embed_dim = 20
    
    if not hasattr(tree_config.model.struct_encoder,"top_k"):
       tree_config.model.struct_encoder.top_k = 30 #default
    
    if not hasattr(tree_config.model.struct_encoder,"gvp_num_layers"):
       tree_config.model.struct_encoder.gvp_num_layers = 3 #default
    
    if not hasattr(tree_config.model.struct_encoder,"use_rotary_embeddings"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_rotary_embeddings=False
    
    if not hasattr(tree_config.model.struct_encoder,"use_foldseek"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek=False
    
    if not hasattr(tree_config.model.struct_encoder,"use_foldseek_vector"): #configs also have num_rbf and num_positional_embeddings
        tree_config.model.struct_encoder.use_foldseek_vector=False
    
    if not hasattr(tree_config.model.struct_encoder,"num_rbf"):
       tree_config.model.struct_encoder.num_rbf = 16 #default
    
    if not hasattr(tree_config.model.struct_encoder,"num_positional_embeddings"):
       tree_config.model.struct_encoder.num_positional_embeddings = 16 #default
    
    if not hasattr(tree_config.model.struct_encoder,"node_h_dim"):
       tree_config.model.struct_encoder.node_h_dim = (100,32) #default
    else:
       tree_config.model.struct_encoder.node_h_dim = ast.literal_eval(tree_config.model.struct_encoder.node_h_dim)
    
    if not hasattr(tree_config.model.struct_encoder,"edge_h_dim"):
       tree_config.model.struct_encoder.edge_h_dim = (32,1) #default
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
