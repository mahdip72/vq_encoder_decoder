import torch
from model import SimpleVQAutoEncoder
from data_test import load_fashion_mnist_data
import cv2


def main():
    # net = SimpleVQAutoEncoder(codebook_size=256)

    train_dataloader = load_fashion_mnist_data(batch_size=1, shuffle=False)

    # Iterate over dataset and the plot each image in a batch using opencv
    for i, (images, labels) in enumerate(train_dataloader):
        print(f"Batch {i} of images has shape {images.shape}")
        print(f"Batch {i} of labels has shape {labels.shape}")
        img = images.squeeze().numpy()
        # resize the image to 256x256
        img = cv2.resize(img, (256, 256))

        cv2.imshow('image', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
