import argparse
import numpy as np
from pathlib import Path
from box import Box
import torch
import cv2
from utils.utils import load_checkpoints_simple
from accelerate import Accelerator
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


def write_codebook(codebook, write_file):
    """
    Write a codebook to a file.
    :param codebook: (torch.Tensor) codebook (2D int array)
    :param write_file: (file) file to which to write the codebook
    :return: None
    """
    for row in codebook:

        for item in row:
            write_file.write(str(item.item()))
            write_file.write(" ")

        write_file.write("\n")

    write_file.write("------------------------------------------")
    write_file.write("\n")


def main(configs):

    checkpoint_dir = configs.checkpoint_dir

    # Load the latest checkpoint in the checkpoint directory
    # Assume the latest checkpoint contains the best model (lowest loss)
    net = prepare_models(configs, None, None)
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    net = load_checkpoints_simple(checkpoint_path, net)
    print('Loaded model from', str(checkpoint_path))

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        dispatch_batches=False
    )

    train_dataloader,test_dataloader = prepare_dataloaders(configs)

    with (torch.inference_mode()):

        # Initialize the progress bar using tqdm
        progress_bar = tqdm(test_dataloader,
                            leave=False,
                            disable=not (accelerator.is_main_process and configs.tqdm_progress_bar))

        # Prepare output txt file for writing
        output_dir = Path(configs.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / Path("inference_contact_codebooks.txt")

        with open(file_path, "w") as write_file:
            for cmaps in progress_bar:
                vq_output, indices, commit_loss = net(cmaps)
                for i in range(len(indices)):
                    codebook = indices[i]
                    # Write codebook to txt file
                    write_codebook(codebook, write_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference on contact maps.")
    parser.add_argument("--config_path", "-c", help="The location of config file", default='configs/inference_config.yaml')

    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)
    main_configs = load_configs(config_file)

    main(main_configs)