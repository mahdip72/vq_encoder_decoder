import numpy as np
from pathlib import Path
import torch
import cv2
from utils.utils import load_checkpoints_simple
from data.data_cifar import prepare_dataloaders
from models.vqvae_cifar import prepare_models
import yaml
from utils.utils import load_configs
from torchvision.transforms import transforms


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

    config_path = "configs/config_cifar.yaml"
    checkpoint_dir = "./results/important_models/2024-06-05__13-28-05/checkpoints/"
    #checkpoint_path = "/home/renjz/vq_encoder_decoder/results/2024-05-31__10-30-51/checkpoints/epoch_32.pth"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    configs = load_configs(config_file)

    # Load the latest checkpoint in the checkpoint directory
    # Assume the latest checkpoint contains the best model (lowest loss)
    net = prepare_models(configs, None, None)
    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    net = load_checkpoints_simple(checkpoint_path, net)
    print('Loaded model from', str(checkpoint_path))

    train_dataloader,test_dataloader = prepare_dataloaders(configs)

    with torch.inference_mode():

        for inputs, labels in test_dataloader:
            vq_output, indices, commit_loss = net(inputs)

            for i in range(len(inputs)):

                # Display the input image
                img_input = inputs[i]
                img_before = img_input.squeeze()
                # img_before = (img_before * 255).astype(np.uint8)
                img_before = normalize_img(img_before).numpy()
                # Move channels from index 0 to index 2
                img_before = np.transpose(img_before, (1, 2, 0))
                img_before = cv2.resize(img_before, (256, 256))
                input_name = 'input ' + str(i)
                # Reverse order of channels (RGB --> BGR)
                cv2.imshow(input_name, img_before[:, :, ::-1])

                # Display the reconstructed output image
                img_output = vq_output[i]
                img_after = img_output.squeeze()
                print(img_after.shape)
                #img_after = np.clip(img_after, 0, 1)
                #img_after = (img_after * 255).astype(np.uint8)
                img_after = normalize_img(img_after).numpy()
                # Move channels from index 0 to index 2
                img_after = np.transpose(img_after, (1, 2, 0))
                img_after = cv2.resize(img_after, (256, 256))
                recons_name = 'recons ' + str(i)
                # Reverse order of channels (RGB --> BGR)
                cv2.imshow('recons ' + str(i), img_after[:, :, ::-1])

                # Set position of images
                cv2.moveWindow(input_name, 100, 50)
                cv2.moveWindow(recons_name, 600, 50)

                # Proceed with the loop when a key is pressed
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()