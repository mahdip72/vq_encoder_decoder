import numpy as np
from pathlib import Path
from box import Box
import torch
import cv2
from utils.utils import load_checkpoints_simple
from data.data_contactmap import prepare_dataloaders
from models.vqvae_contact import prepare_models
import yaml
from utils.utils import load_configs
from torchvision.transforms import transforms
from tqdm import tqdm


def normalize_img(image):
    """
    Normalize image from [-1,1] to [0,255]
    :param image: (tensor) image to transform
    :return: (tensor) normalized image
    """
    transform = transforms.Compose([
        transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))  # Normalize the images
    ])
    return (transform(image) * 255).type(torch.uint8)


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


def main():

    config_path = "configs/inference_config.yaml"
    with open(config_path) as file:
        config_file = yaml.full_load(file)
    configs = load_configs(config_file)

    checkpoint_dir = configs.checkpoint_dir

    # Load the latest checkpoint in the checkpoint directory
    # Assume the latest checkpoint contains the best model (lowest loss)
    net = prepare_models(configs, None, None)
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    net = load_checkpoints_simple(checkpoint_path, net)
    print('Loaded model from', str(checkpoint_path))

    train_dataloader,test_dataloader = prepare_dataloaders(configs)

    with (torch.inference_mode()):

        # Initialize the progress bar using tqdm
        progress_bar = tqdm(test_dataloader,
                            leave=False,
                            disable=not (configs.tqdm_progress_bar))
        #disable = not (accelerator.is_main_process and configs.tqdm_progress_bar))

        for cmaps in progress_bar:
            vq_output, indices, commit_loss = net(cmaps)
            for cmap in cmaps:

                pass
                # Save codebooks


if __name__ == '__main__':
    main()