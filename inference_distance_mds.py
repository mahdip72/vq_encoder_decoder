import argparse
import numpy as np
from pathlib import Path
import torch
from utils.utils import load_checkpoints_simple
from accelerate import Accelerator
from data.dataset import prepare_distance_map_vqvae_dataloaders
from models.distance_map_vqvae import prepare_models_distance_map_vqvae
import yaml
from utils.utils import load_configs
from tqdm import tqdm


def get_latest_checkpoint(checkpoint_dir):
    """
    Determine the latest checkpoint in a directory of checkpoints.
    Assume the checkpoint files follow this format: 'epoch_#.pth', where #
    represents the epoch number.
    :param checkpoint_dir: (str or Path) path to directory of checkpoints
    """
    max_epoch = 0
    latest_checkpoint = None

    # Iterate through the checkpoint directory and find the checkpoint
    # with the most recent epoch
    for checkpoint_file in Path(checkpoint_dir).glob('*.pth'):
        file_stem = str(checkpoint_file.stem)
        epoch_num = int(file_stem.split('_')[1])

        if epoch_num > max_epoch:
            max_epoch = epoch_num
            latest_checkpoint = checkpoint_file

    return latest_checkpoint


def main(configs):

    checkpoint_dir = configs.checkpoint_dir

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        dispatch_batches=False
    )

    # Load the latest checkpoint in the checkpoint directory
    # Assume the latest checkpoint contains the best model (lowest loss)
    net = prepare_models_distance_map_vqvae(configs, None, accelerator)
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    net = load_checkpoints_simple(checkpoint_path, net)
    print('Loaded model from', str(checkpoint_path))

    net.to(accelerator.device)

    train_dataloader,test_dataloader, vis_dataloader = prepare_distance_map_vqvae_dataloaders(configs)

    with (torch.inference_mode()):

        # Initialize the progress bar using tqdm
        progress_bar = tqdm(test_dataloader,
                            leave=False,
                            disable=not (accelerator.is_main_process and configs.tqdm_progress_bar))

        for cmaps in progress_bar:
            cmaps = cmaps.to(accelerator.device)
            vq_output, indices, commit_loss = net(cmaps)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference on contact maps.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='configs/config_distance_mds.yaml')

    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)
    main_configs = load_configs(config_file)

    main(main_configs)