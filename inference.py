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


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = SimpleVQAutoEncoder(codebook_size=256)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def inference(model, data_loader):
    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for x, _ in data_loader:
            reconstructed, indices, _ = model(x)
            # Optionally, visualize the results
            visualize_results(x.cpu(), reconstructed.cpu())


def visualize_results(original, reconstructed):
    num_images = min(len(original), 10)  # Show up to 10 images
    original_concat = np.concatenate([original[i].squeeze(0).numpy() for i in range(num_images)], axis=1)
    reconstructed_concat = np.concatenate([reconstructed[i].squeeze(0).numpy() for i in range(num_images)], axis=1)
    combined = np.concatenate((original_concat, reconstructed_concat), axis=0)

    combined = ((combined + 1) * 0.5 * 255).astype(np.uint8)

    cv2.imshow('Original (top) and Reconstructed (bottom) Images', combined)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def visualize_vq_outputs_cv2(vq_outputs, indices, commit_losses):
    batch_size = vq_outputs.shape[0]
    scale_factor = 64  # Scale factor for viewing the images larger

    for i in range(batch_size):
        # Feature maps visualization
        feature_map = vq_outputs[i].cpu().detach().numpy()
        if feature_map.ndim > 2:
            feature_map = np.mean(feature_map, axis=0)  # Average over channels if any

        feature_map = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
        feature_map = cv2.applyColorMap(feature_map.astype(np.uint8), cv2.COLORMAP_JET)
        feature_map = cv2.resize(feature_map,
                                 (feature_map.shape[1] * scale_factor, feature_map.shape[0] * scale_factor),
                                 interpolation=cv2.INTER_NEAREST)

        # Indices visualization
        index_image = indices[i].cpu().detach().numpy().astype(np.uint8)
        index_image = cv2.resize(index_image,
                                 (index_image.shape[1] * scale_factor, index_image.shape[0] * scale_factor),
                                 interpolation=cv2.INTER_NEAREST)
        index_image = cv2.applyColorMap(index_image, cv2.COLORMAP_HSV)

        # Concatenate images horizontally
        combined_image = cv2.hconcat([feature_map, index_image])

        # Display the image with commitment loss in the window title
        window_title = f"Sample {i + 1} - Commitment Loss: {commit_losses[i].item():.4f}"
        cv2.imshow(window_title, combined_image)
        cv2.waitKey(0)  # Wait for a key press to continue to the next image

    cv2.destroyAllWindows()


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
