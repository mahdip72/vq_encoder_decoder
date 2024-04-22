import torch
from model import SimpleVQAutoEncoder
from data_test import load_fashion_mnist_data
import cv2
import numpy as np
from utils import *
from accelerate import Accelerator
from data_test import *
from model import SimpleVQAutoEncoder
from tqdm import tqdm


def load_model(checkpoint_path='epoch_8.pth'):
    model = SimpleVQAutoEncoder(codebook_size=256)
    load_checkpoints_simple(checkpoint_path, model)
    return model






def main():
    net = SimpleVQAutoEncoder(codebook_size=256)

    test_dataloader = load_fashion_mnist_data(batch_size=1, shuffle=False)

    with torch.inference_mode():
        for inputs in test_dataloader:
                img_before = inputs[0].squeeze().numpy()
                img_before = cv2.resize(img_before, (256, 256))
                cv2.imshow('image', img_before)
                cv2.waitKey(0)

                vq_output, indices, commit_loss = net(inputs[0], return_vq_only=True)

                img_after = vq_output[0, 0]  # Shape (7, 7)
                img_after = (img_after * 255).to(torch.uint8).numpy()
                img_after = cv2.resize(img_after, (256, 256), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Image", img_after)
                cv2.waitKey(0)




if __name__ == '__main__':
    main()
