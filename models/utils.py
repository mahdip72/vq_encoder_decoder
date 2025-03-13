import torch_geometric
import torch


def separate_features(batched_features, batch):
    # Split the features tensor into separate tensors for each graph
    features_list = torch_geometric.utils.unbatch(batched_features, batch)
    return features_list


def merge_features(features_list, max_length):
    # Pad tensors and create masks
    device = features_list[0].device

    padded_tensors = []
    masks = []
    slice_lengths = [0]
    for t in features_list:
        # Create mask of size (original_length,)
        mask = torch.ones(t.size(0), device=t.device)

        if t.size(0) < max_length:
            size_diff = max_length - t.size(0)
            pad = torch.zeros(size_diff, t.size(1), device=t.device)
            t_padded = torch.cat([t, pad], dim=0)

            # Pad mask with zeros for the padded positions
            mask = torch.cat([mask, torch.zeros(size_diff, device=t.device)], dim=0)
        else:
            t_padded = t[:max_length, :]
            mask = mask[:max_length]  # Trim mask if necessary

        padded_tensors.append(t_padded.unsqueeze(0))  # Add an extra dimension for concatenation
        masks.append(mask.unsqueeze(0))  # Add an extra dimension to mask as well
        slice_lengths.append(min(max_length, t.size(0)))

    # Concatenate tensors and masks
    padded_features = torch.cat(padded_tensors, dim=0)
    mask = torch.cat(masks, dim=0).bool()

    # Flatten padded features and mask
    flat_mask = mask.view(-1)  # Shape: (num_batches * max_length)

    # Create batch assignment tensor
    num_batches = padded_features.size(0)
    batch_indices = torch.arange(num_batches, device=device).unsqueeze(1).repeat(1, max_length).view(
        -1)  # Shape: (num_batches * max_length)
    valid_batch_indices = batch_indices[flat_mask.bool()]  # Filter based on mask

    slice_indices = torch.cumsum(torch.tensor(slice_lengths), dim=0)
    slice_indices[-1] -= 1  # Decrement the last element

    return padded_features, mask, valid_batch_indices, slice_indices