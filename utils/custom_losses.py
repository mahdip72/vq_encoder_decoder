import torch
import torch.nn.functional as F
from graphein.protein.tensor.geometry import quaternion_to_matrix
from utils.alignment import kabsch


def compute_grad_norm(loss, parameters, norm_type=2):
    """
    Compute the gradient norm for a given loss and model parameters without altering existing gradients.

    Args:
        loss (torch.Tensor): The loss tensor.
        parameters (iterable): Iterable of model parameters.
        norm_type (float): The type of norm (default 2 for L2 norm).

    Returns:
        torch.Tensor: The gradient norm.
    """
    grads = torch.autograd.grad(
        loss,
        [p for p in parameters if p.requires_grad],
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    grads = [g for g in grads if g is not None]
    if not grads:
        return torch.tensor(0.0)
    norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return norm


def quaternion_align(P, Q, weights=None):
    """
    Quaternion-based alignment of point sets (Horn's method).
    Aligns Q to P (true coordinates to predicted coordinates).

    Args:
        P: Predicted coordinates [batch, num_points, 3]
        Q: True coordinates [batch, num_points, 3]
        weights: Optional weights for each point [batch, num_points]

    Returns:
        R: Rotation matrices [batch, 3, 3]
        t: Translation vectors [batch, 3, 1]
    """
    batch_size = P.shape[0]
    device = P.device

    if weights is None:
        weights = torch.ones(P.shape[0], P.shape[1], device=device)

    # Normalize weights to sum to 1 per batch
    weights_sum = weights.sum(dim=1, keepdim=True)
    norm_weights = weights / (weights_sum + 1e-8)

    # Calculate centroids
    weights_expanded = norm_weights.unsqueeze(-1)
    p_center = torch.sum(P * weights_expanded, dim=1, keepdim=True)
    q_center = torch.sum(Q * weights_expanded, dim=1, keepdim=True)

    # Center the coordinates
    p_centered = P - p_center
    q_centered = Q - q_center

    # Compute weighted covariance matrix
    C = torch.bmm(p_centered.transpose(1, 2), q_centered * weights_expanded)

    # Construct the quaternion matrix
    K = torch.zeros(batch_size, 4, 4, device=device)

    K[:, 0, 0] = C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2]
    K[:, 0, 1] = C[:, 1, 2] - C[:, 2, 1]
    K[:, 0, 2] = C[:, 2, 0] - C[:, 0, 2]
    K[:, 0, 3] = C[:, 0, 1] - C[:, 1, 0]

    K[:, 1, 0] = C[:, 1, 2] - C[:, 2, 1]
    K[:, 1, 1] = C[:, 0, 0] - C[:, 1, 1] - C[:, 2, 2]
    K[:, 1, 2] = C[:, 0, 1] + C[:, 1, 0]
    K[:, 1, 3] = C[:, 0, 2] + C[:, 2, 0]

    K[:, 2, 0] = C[:, 2, 0] - C[:, 0, 2]
    K[:, 2, 1] = C[:, 0, 1] + C[:, 1, 0]
    K[:, 2, 2] = -C[:, 0, 0] + C[:, 1, 1] - C[:, 2, 2]
    K[:, 2, 3] = C[:, 1, 2] + C[:, 2, 1]

    K[:, 3, 0] = C[:, 0, 1] - C[:, 1, 0]
    K[:, 3, 1] = C[:, 0, 2] + C[:, 2, 0]
    K[:, 3, 2] = C[:, 1, 2] + C[:, 2, 1]
    K[:, 3, 3] = -C[:, 0, 0] - C[:, 1, 1] + C[:, 2, 2]

    # Find eigenvector with largest eigenvalue
    _, eigenvecs = torch.linalg.eigh(K)
    q = eigenvecs[:, :, -1]  # Last column is the eigenvector for largest eigenvalue

    # Convert quaternion to rotation matrix
    R = quaternion_to_matrix(q)

    # Calculate translation
    t = p_center.transpose(1, 2) - torch.bmm(R, q_center.transpose(1, 2))  # Ensure t is [batch, 3, 1]

    return R, t


def compute_quaternion_alignment(x_pred, x_tru, mask):
    # Flatten valid coordinates along sequence and atom dimensions for quaternion alignment
    pred_flat = x_pred[mask].reshape(-1, 3)  # Shape: [num_valid_points, 3]
    true_flat = x_tru[mask].reshape(-1, 3)  # Shape: [num_valid_points, 3]

    # Create batch dimension of 1 for the quaternion_align function
    pred_batch = pred_flat.unsqueeze(0)  # Shape: [1, num_valid_points, 3]
    true_batch = true_flat.unsqueeze(0)  # Shape: [1, num_valid_points, 3]

    # Compute rotation and translation to align true to predicted coordinates
    # Note: quaternion_align aligns Q to P (true to predicted)
    R, t = quaternion_align(pred_batch, true_batch)

    # Apply rotation and translation to all true coordinates (not just masked ones)
    # First reshape to batch dim for matrix multiplication
    x_tru_reshaped = x_tru.reshape(-1, 3).unsqueeze(0)  # Shape: [1, seq_len*num_atoms, 3]

    # Apply rotation
    rotated = torch.bmm(R, x_tru_reshaped.transpose(1, 2)).transpose(1, 2)  # Shape: [1, seq_len*num_atoms, 3]

    # Apply translation - Fix the shape of t before broadcasting
    # Ensure t has correct shape for broadcasting
    if t.dim() == 3 and t.shape[1] == 3 and t.shape[2] != 1:
        # If t is [1, 3, 3] or similar, extract the translation vector
        t = t[:, :, 0].unsqueeze(2)  # Shape: [1, 3, 1]

    # Now properly reshape for broadcasting
    t = t.transpose(1, 2)  # Shape: [1, 1, 3]
    # Expand t to match the shape of rotated
    t_broadcasted = t.expand_as(rotated)  # Shape: [1, seq_len*num_atoms, 3]

    # Add the translation to the rotated coordinates
    x_true_aligned = (rotated + t_broadcasted).squeeze(0).reshape_as(x_tru)  # Shape: [seq_len, num_atoms, 3]

    return x_true_aligned


def calculate_aligned_mse_loss(x_predicted, x_true, masks, alignment_strategy):
    """
    Calculates the MSE loss between x_predicted and x_true after performing alignment,
    applying the provided masks.

    Parameters:
    x_predicted (torch.Tensor): Predicted coordinates of shape [batch_size, seq_len, num_atoms, 3].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.
    alignment_strategy (str): Strategy for alignment. Options: 'kabsch', 'kabsch_old', 'quaternion', 'no'.
                              Use 'no' for no alignment (absolute position error).

    Returns:
    torch.Tensor: The computed MSE loss for each batch element.
    torch.Tensor: The predicted coordinates.
    torch.Tensor: The aligned true coordinates.
    """
    batch_size = x_predicted.shape[0]
    loss_list = []
    x_true_aligned_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_pred = x_predicted[i]  # [seq_len, num_atoms, 3]
        x_tru = x_true[i]  # [seq_len, num_atoms, 3]

        with torch.no_grad():
            if alignment_strategy == 'kabsch':
                # Extract valid residues (each with multiple atoms) based on mask
                x_tru_valid = x_tru[mask]  # Shape: (num_valid_residues, num_atoms, 3)
                x_pred_valid = x_pred[mask]
                # Clone full true coords to fill in aligned residues
                x_true_aligned = x_tru.clone()
                # Flatten atoms across residues for Kabsch input
                coords_true_flat = x_tru_valid.reshape(-1, 3)
                coords_pred_flat = x_pred_valid.reshape(-1, 3)
                # Perform Kabsch alignment on flattened points
                aligned_flat = kabsch(
                    coords_true_flat, coords_pred_flat,
                    ca_only=False,  # Use all backbone atoms for alignment
                    allow_reflections=False,
                    return_transformed=True
                ).detach()
                # Reshape back to per-residue atom layout
                aligned_valid = aligned_flat.reshape_as(x_tru_valid)
                # Assign aligned residues back into the full tensor
                x_true_aligned[mask] = aligned_valid

            elif alignment_strategy == 'quaternion':
                x_true_aligned = compute_quaternion_alignment(x_pred, x_tru, mask).detach()

            elif alignment_strategy == 'no':
                # No alignment, use original true coordinates directly
                x_true_aligned = x_tru.detach()

            x_true_aligned_list.append(x_true_aligned)

        # Compute MSE loss using the masked areas
        loss = torch.mean(((x_pred[mask] - x_true_aligned[mask]) ** 2))
        loss_list.append(loss)

    return torch.stack(loss_list), x_predicted, torch.stack(x_true_aligned_list)


def calculate_backbone_distance_loss(x_predicted, x_true, masks):
    """
    Calculates the backbone distance loss between x_predicted and x_true after applying masks.

    Parameters:
    x_predicted (torch.Tensor): Predicted coordinates of shape [batch_size, seq_len, num_atoms, 3].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed backbone distance loss for each batch element.
    """
    batch_size = x_predicted.shape[0]
    loss_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_pred = x_predicted[i][mask]  # [num_valid, num_atoms, 3]
        x_tru = x_true[i][mask]

        # Extract the backbone atoms (N, CA, C)
        x_pred_backbone = x_pred[:, :3, :].reshape(-1, 3)
        x_tru_backbone = x_tru[:, :3, :].reshape(-1, 3)

        # Compute pairwise L2 distance matrices
        D_pred = torch.cdist(x_pred_backbone, x_pred_backbone, p=2)
        D_true = torch.cdist(x_tru_backbone, x_tru_backbone, p=2)

        # Compute the squared differences
        distance_diff = (D_pred - D_true) ** 2

        # Clamp the maximum error to (5 Å)^2
        clamped_diff = torch.clamp(distance_diff, max=5 ** 2)

        # Take the mean of the clamped differences
        loss = clamped_diff.mean()
        loss_list.append(loss)

    return torch.stack(loss_list)


def compute_vectors(coords):
    """
    Computes the six vectors described in the pseudocode.

    Args:
        coords (torch.Tensor): The input coordinates of shape (num_points, num_atoms, 3).

    Returns:
        torch.Tensor: The computed vectors of shape (6 * num_points, 3).
    """
    # Compute the vectors
    N_to_CA = coords[:, 1, :] - coords[:, 0, :]
    CA_to_C = coords[:, 2, :] - coords[:, 1, :]
    C_to_Nnext = F.pad(
        coords[1:, 0, :] - coords[:-1, 2, :],
        (0, 0, 0, 1),
        value=0.0
    )

    Cprev_to_N = F.pad(
        coords[1:, 0, :] - coords[:-1, 2, :],
        (0, 0, 1, 0),
        value=0.0
    )

    # Compute the normal vectors
    nCA = -torch.cross(N_to_CA, CA_to_C, dim=-1)
    nN = torch.cross(Cprev_to_N, N_to_CA, dim=-1)
    nC = torch.cross(CA_to_C, C_to_Nnext, dim=-1)

    # Concatenate all vectors
    vectors = torch.cat([N_to_CA, CA_to_C, C_to_Nnext, nCA, nN, nC], dim=0)

    return vectors


def calculate_backbone_direction_loss(x_predicted, x_true, masks):
    """
    Calculates the backbone direction loss between x_predicted and x_true after applying masks.

    Parameters:
    x_predicted (torch.Tensor): Predicted coordinates of shape [batch_size, seq_len, num_atoms, 3].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed backbone direction loss for each batch element.
    """
    batch_size = x_predicted.shape[0]
    loss_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_pred = x_predicted[i][mask]  # [num_valid, num_atoms, 3]
        x_tru = x_true[i][mask]

        # Compute vectors for predicted and true coordinates
        V_pred = compute_vectors(x_pred)
        V_true = compute_vectors(x_tru)

        # Compute pairwise dot products
        D_pred = torch.matmul(V_pred, V_pred.transpose(0, 1))
        D_true = torch.matmul(V_true, V_true.transpose(0, 1))

        # Compute squared differences
        E = (D_pred - D_true) ** 2

        # Clamp the maximum error to 20
        E = torch.clamp(E, max=20)

        # Take the mean of the clamped differences
        loss = E.mean()
        loss_list.append(loss)

    return torch.stack(loss_list)


def calculate_binned_direction_classification_loss(dir_loss_logits, x_true, masks):
    """
    Calculates the binned direction classification loss.

    Parameters:
    dir_loss_logits (torch.Tensor): Logits of shape [batch_size, seq_len, seq_len, 6, 16].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed binned direction classification loss for each batch element.
    """
    batch_size, _, _, _ = x_true.shape
    loss_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_tru = x_true[i][mask]  # [num_valid, num_atoms, 3]

        # Compute vectors for true coordinates
        CA_to_C = x_tru[:, 2, :] - x_tru[:, 1, :]
        CA_to_N = x_tru[:, 0, :] - x_tru[:, 1, :]
        nCA = torch.cross(CA_to_C, CA_to_N, dim=-1)

        # Normalize vectors to unit length
        CA_to_C = F.normalize(CA_to_C, dim=-1)
        CA_to_N = F.normalize(CA_to_N, dim=-1)
        nCA = F.normalize(nCA, dim=-1)

        # Compute pairwise dot products
        dot_products = torch.stack([
            torch.matmul(CA_to_N, CA_to_N.transpose(0, 1)),
            torch.matmul(CA_to_N, CA_to_C.transpose(0, 1)),
            torch.matmul(CA_to_N, nCA.transpose(0, 1)),
            torch.matmul(CA_to_C, CA_to_C.transpose(0, 1)),
            torch.matmul(CA_to_C, nCA.transpose(0, 1)),
            torch.matmul(nCA, nCA.transpose(0, 1))
        ], dim=-1)  # Shape: [num_valid, num_valid, 6]

        # Bin the dot products into 16 bins
        bins = torch.linspace(-1, 1, 16, device=x_true.device)
        dot_products = torch.clamp(dot_products, min=-0.9999, max=0.9999)
        labels = torch.bucketize(dot_products, bins)  # Shape: [num_valid, num_valid, 6]

        # Compute cross-entropy loss
        logits = dir_loss_logits[i][mask][:, mask]  # Shape: [num_valid, num_valid, 6, 16]
        loss = F.cross_entropy(logits.view(-1, 16), labels.view(-1), reduction='mean')
        loss_list.append(loss)

    return torch.stack(loss_list)


def calculate_binned_distance_classification_loss(dist_loss_logits, x_true, masks):
    """
    Calculates the binned distance classification loss.

    Parameters:
    dist_loss_logits (torch.Tensor): Logits of shape [batch_size, seq_len, seq_len, 64].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed binned distance classification loss for each batch element.
    """
    batch_size, _, _, _ = x_true.shape
    loss_list = []

    # Define the bin edges
    bin_edges = torch.tensor([(2.3125 + 0.3075 * i) ** 2 for i in range(63)], device=x_true.device)

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        x_tru = x_true[i][mask]  # [num_valid, num_atoms, 3]

        # Compute Cβ coordinates
        N_to_CA = x_tru[:, 1, :] - x_tru[:, 0, :]
        CA_to_C = x_tru[:, 2, :] - x_tru[:, 1, :]
        n = torch.cross(N_to_CA, CA_to_C, dim=-1)

        a = -0.58273431
        b = 0.56802827
        c = -0.54067466

        C_beta = a * n + b * N_to_CA + c * CA_to_C + x_tru[:, 1, :]

        # Compute pairwise distances
        pairwise_distances = torch.cdist(C_beta, C_beta, p=2) ** 2

        # Bin the distances into 64 bins
        labels = torch.bucketize(pairwise_distances, bin_edges)  # Shape: [num_valid, num_valid]

        # Compute cross-entropy loss
        logits = dist_loss_logits[i][mask][:, mask]  # Shape: [num_valid, num_valid, 64]
        loss = F.cross_entropy(logits.view(-1, 64), labels.view(-1), reduction='mean')
        loss_list.append(loss)

    return torch.stack(loss_list)


def calculate_decoder_loss(x_predicted, x_true, masks, configs, seq=None, dir_loss_logits=None, dist_loss_logits=None,
                           alignment_strategy='kabsch'):
    # Compute aligned MSE foundation
    mse_raw, x_pred_aligned, x_true_aligned = calculate_aligned_mse_loss(
        x_predicted, x_true, masks, alignment_strategy=alignment_strategy)
    device = x_predicted.device

    # Prepare loss dict with weighted components or 0.0 if disabled
    loss_dict = {}
    # MSE reconstruction
    if configs.train_settings.losses.mse.enabled:
        w = configs.train_settings.losses.mse.weight
        loss_dict['mse_loss'] = mse_raw.mean() * w
    else:
        loss_dict['mse_loss'] = torch.tensor(0.0, device=device)
    # Backbone distance
    if configs.train_settings.losses.backbone_distance.enabled:
        w = configs.train_settings.losses.backbone_distance.weight
        loss_dict['backbone_distance_loss'] = calculate_backbone_distance_loss(
            x_pred_aligned, x_true_aligned, masks).mean() * w
    else:
        loss_dict['backbone_distance_loss'] = torch.tensor(0.0, device=device)
    # Backbone direction
    if configs.train_settings.losses.backbone_direction.enabled:
        w = configs.train_settings.losses.backbone_direction.weight
        loss_dict['backbone_direction_loss'] = calculate_backbone_direction_loss(
            x_pred_aligned, x_true_aligned, masks).mean() * w
    else:
        loss_dict['backbone_direction_loss'] = torch.tensor(0.0, device=device)
    # Binned direction classification
    if configs.train_settings.losses.binned_direction_classification.enabled:
        w = configs.train_settings.losses.binned_direction_classification.weight
        val = calculate_binned_direction_classification_loss(
            dir_loss_logits, x_true_aligned, masks).mean() if dir_loss_logits is not None else torch.tensor(0.0,
                                                                                                            device=device)
        loss_dict['binned_direction_classification_loss'] = val * w
    else:
        loss_dict['binned_direction_classification_loss'] = torch.tensor(0.0, device=device)
    # Binned distance classification
    if configs.train_settings.losses.binned_distance_classification.enabled:
        w = configs.train_settings.losses.binned_distance_classification.weight
        val = calculate_binned_distance_classification_loss(
            dist_loss_logits, x_true_aligned, masks).mean() if dist_loss_logits is not None else torch.tensor(0.0,
                                                                                                              device=device)
        loss_dict['binned_distance_classification_loss'] = val * w
    else:
        loss_dict['binned_distance_classification_loss'] = torch.tensor(0.0, device=device)

    # Sum reconstruction components
    valid_losses = [v for k, v in loss_dict.items() if 'loss' in k and k != 'rec_loss' and not torch.isnan(v)]
    if not valid_losses:
        loss_dict['rec_loss'] = torch.tensor(0.0, device=device)
    else:
        loss_dict['rec_loss'] = sum(valid_losses)
    return loss_dict, x_pred_aligned, x_true_aligned

