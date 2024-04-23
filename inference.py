import torch
import cv2
from utils import load_checkpoints_simple
from data_test import load_fashion_mnist_data
from model import SimpleVQAutoEncoder


def main():
    import yaml
    from utils import load_configs

    config_path = "results/2024-04-22__20-20-11/config.yaml"
    checkpoint_path = "results/2024-04-22__20-20-11/checkpoints/epoch_8.pth"

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

    test_dataloader = load_fashion_mnist_data(batch_size=1, shuffle=False)

    with torch.inference_mode():
        for inputs in test_dataloader:
            img_before = inputs[0].squeeze().numpy()
            img_before = cv2.resize(img_before, (256, 256))
            cv2.imshow('input', img_before)

            vq_output, indices, commit_loss = net(inputs[0], return_vq_only=True)

            img_after = vq_output[0, 0]  # Shape (7, 7)
            img_after = (img_after * 255).to(torch.uint8).numpy()
            img_after = cv2.resize(img_after, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("recons", img_after)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
