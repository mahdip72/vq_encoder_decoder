import numpy as np
import torch
import cv2
from utils import load_checkpoints_simple
from data_test import load_fashion_mnist_data, load_cifar10_data
from model import SimpleVQAutoEncoder


def main():
    import yaml
    from utils import load_configs

    config_path = "results/2024-04-22__21-53-27/config.yaml"
    checkpoint_path = "results/2024-04-22__21-53-27/checkpoints/epoch_24.pth"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    configs = load_configs(config_file)

    net = SimpleVQAutoEncoder(
        dim=configs.model.vector_quantization.dim,
        codebook_size=configs.model.vector_quantization.codebook_size,
        decay=configs.model.vector_quantization.decay,
        commitment_weight=configs.model.vector_quantization.commitment_weight
    )

    net = load_checkpoints_simple(checkpoint_path, net)

    # test_dataloader = load_fashion_mnist_data(batch_size=1, shuffle=False)
    test_dataloader = load_cifar10_data(batch_size=1, shuffle=False)

    with torch.inference_mode():
        for inputs in test_dataloader:
            img_before = inputs[0].squeeze().numpy()
            img_before = (img_before * 255).astype(np.uint8)
            img_before = np.transpose(img_before, (1, 2, 0))
            img_before = cv2.resize(img_before, (256, 256))
            cv2.imshow('input', img_before[:, :, ::-1])

            vq_output, indices, commit_loss = net(inputs[0], return_vq_only=False)

            img_after = vq_output.squeeze().numpy()
            img_after = np.clip(img_after, 0, 1)
            img_after = (img_after * 255).astype(np.uint8)
            img_after = np.transpose(img_after, (1, 2, 0))
            img_after = cv2.resize(img_after, (256, 256))
            cv2.imshow("recons", img_after[:, :, ::-1])
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
