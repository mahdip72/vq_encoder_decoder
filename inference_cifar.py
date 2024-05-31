import numpy as np
import torch
import cv2
from utils.utils import load_checkpoints_simple
from data.data_cifar import prepare_dataloaders
from models.vqvae_cifar import prepare_models
import yaml
from utils.utils import load_configs


def main():

    config_path = "configs/config_cifar.yaml"
    checkpoint_path = "./results/2024-05-31__10-30-51/checkpoints/epoch_32.pth"
    #checkpoint_path = "/home/renjz/vq_encoder_decoder/results/2024-05-31__10-30-51/checkpoints/epoch_32.pth"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    configs = load_configs(config_file)

    net = prepare_models(configs, None, None)

    net = load_checkpoints_simple(checkpoint_path, net)

    train_dataloader,test_dataloader = prepare_dataloaders(configs)

    with torch.inference_mode():
        for inputs, labels in test_dataloader:
            img_before = inputs.squeeze().numpy()
            img_before = (img_before * 255).astype(np.uint8)
            img_before = np.transpose(img_before, (1, 2, 0))
            img_before = cv2.resize(img_before, (256, 256))
            cv2.imshow('input', img_before[:, :, ::-1])

            vq_output, indices, commit_loss = net(inputs)

            img_after = vq_output.squeeze().numpy()
            img_after = np.clip(img_after, 0, 1)
            img_after = (img_after * 255).astype(np.uint8)
            img_after = np.transpose(img_after, (1, 2, 0))
            img_after = cv2.resize(img_after, (256, 256))
            cv2.imshow("recons", img_after[:, :, ::-1])
            cv2.waitKey(0)


if __name__ == '__main__':
    main()