import torch_geometric
import torch
import time


def separate_features(batched_features, batch):
    # Split the features tensor into separate tensors for each graph
    features_list = torch_geometric.utils.unbatch(batched_features, batch)
    return features_list


def merge_features(features_list, max_length):
    """
    Merges a list of feature tensors into a single padded tensor with a mask.

    Args:
        features_list (list[torch.Tensor]): A list of tensors, where each tensor
                                             has shape (seq_len, feature_dim).
        max_length (int): The maximum sequence length to pad or truncate to.

    Returns:
        tuple:
            - padded_features (torch.Tensor): Padded/truncated features of shape
                                              (num_batches, max_length, feature_dim).
            - mask (torch.Tensor): Boolean mask of shape (num_batches, max_length),
                                   True for valid positions.
    """
    num_batches = len(features_list)

    if num_batches == 0:
        # Handle empty input list gracefully
        return (torch.empty((0, max_length, 0)),  # padded_features
                torch.empty((0, max_length), dtype=torch.bool))  # mask

    # Get properties from the first tensor (assuming non-empty list)
    first_tensor = features_list[0]
    device = first_tensor.device
    dtype = first_tensor.dtype
    feature_dim = first_tensor.size(1)

    # Pre-allocate tensors for efficiency
    padded_features = torch.zeros(num_batches, max_length, feature_dim, device=device, dtype=dtype)
    mask = torch.zeros(num_batches, max_length, dtype=torch.bool, device=device)

    # Fill pre-allocated tensors directly using slicing
    for i, t in enumerate(features_list):
        length = min(t.size(0), max_length)
        padded_features[i, :length] = t[:length]
        mask[i, :length] = True

    return padded_features, mask


def test_merge_features_speed(num_batches=128, avg_seq_len=500, feature_dim=512, max_length=1024, num_iterations=100, device='cuda'):
    """
    Tests the speed of the merge_features function on a specified device (GPU recommended).

    Args:
        num_batches (int): Number of tensors in the input list.
        avg_seq_len (int): Average sequence length for generated tensors.
        feature_dim (int): Feature dimension of the tensors.
        max_length (int): Maximum sequence length for padding/truncation.
        num_iterations (int): Number of times to run the function for timing.
        device (str): The device to run the test on ('cuda' or 'cpu').
    """
    print(f"\nTesting merge_features speed on {device}...")
    print(f"Parameters: num_batches={num_batches}, avg_seq_len={avg_seq_len}, feature_dim={feature_dim}, max_length={max_length}, iterations={num_iterations}")

    # Generate random input data on the target device
    features_list = []
    for _ in range(num_batches):
        # Vary sequence lengths slightly around the average
        seq_len = max(1, int(avg_seq_len + torch.randn(1).item() * (avg_seq_len / 4)))
        features_list.append(torch.randn(seq_len, feature_dim, device=device))

    # Warm-up run
    _ = merge_features(features_list, max_length)

    if device == 'cuda':
        # Use CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            _ = merge_features(features_list, max_length)
        end_event.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / num_iterations
    else:
        # Use time.time() for CPU timing
        start_time = time.time()
        for _ in range(num_iterations):
            _ = merge_features(features_list, max_length)
        end_time = time.time()
        elapsed_time_s = end_time - start_time
        avg_time_ms = (elapsed_time_s / num_iterations) * 1000

    print(f"Average execution time over {num_iterations} iterations: {avg_time_ms:.4f} ms")


if __name__ == '__main__':
    # Check if CUDA is available
    if torch.cuda.is_available():
        test_merge_features_speed(
            num_batches=256,
            avg_seq_len=600,
            feature_dim=768,
            max_length=1024,
            num_iterations=200,
            device='cuda'
        )
    else:
        print("\nCUDA not available. Skipping GPU speed test for merge_features.")

    # Optionally, run the CPU test as well or instead
    # test_merge_features_speed(
    #     num_batches=64,
    #     avg_seq_len=500,
    #     feature_dim=512,
    #     max_length=1024,
    #     num_iterations=50,
    #     device='cpu'
    # )
