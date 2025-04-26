import torch_geometric
import torch


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
            - valid_batch_indices (torch.Tensor): Batch index for each valid element
                                                  after flattening, shape (num_valid_elements,).
            - slice_indices (torch.Tensor): Cumulative sum of sequence lengths,
                                            used for potential slicing later.
    """
    num_batches = len(features_list)

    if num_batches == 0:
        # Handle empty input list gracefully
        return (torch.empty((0, max_length, 0)),  # padded_features
                torch.empty((0, max_length), dtype=torch.bool),  # mask
                torch.empty((0,), dtype=torch.long),  # valid_batch_indices
                torch.tensor([0], dtype=torch.long))  # slice_indices

    # Get properties from the first tensor (assuming non-empty list)
    first_tensor = features_list[0]
    device = first_tensor.device
    dtype = first_tensor.dtype
    feature_dim = first_tensor.size(1)

    # Pre-allocate tensors for efficiency
    padded_features = torch.zeros(num_batches, max_length, feature_dim, device=device, dtype=dtype)
    mask = torch.zeros(num_batches, max_length, dtype=torch.bool, device=device)
    actual_lengths = torch.empty(num_batches, dtype=torch.long, device=device)

    # Fill pre-allocated tensors directly using slicing
    for i, t in enumerate(features_list):
        length = min(t.size(0), max_length)
        padded_features[i, :length] = t[:length]
        mask[i, :length] = True
        actual_lengths[i] = length

    # Calculate slice_indices (cumulative lengths) efficiently
    # Prepend a zero for the cumulative sum start
    slice_lengths_for_cumsum = torch.cat((torch.tensor([0], device=device), actual_lengths))
    slice_indices = torch.cumsum(slice_lengths_for_cumsum, dim=0)
    # Decrement the last element to match original behavior (if needed)
    if slice_indices.numel() > 1: # Ensure there's more than just the initial zero
         slice_indices[-1] -= 1

    # Calculate valid_batch_indices efficiently using the final mask
    # nonzero returns indices where mask is True. We need the batch index (dim 0).
    valid_indices_tuple = mask.nonzero(as_tuple=True)
    valid_batch_indices = valid_indices_tuple[0] # Get the row indices (batch indices)

    return padded_features, mask, valid_batch_indices, slice_indices
