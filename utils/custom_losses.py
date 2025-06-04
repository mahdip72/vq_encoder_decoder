import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from graphein.protein.tensor.geometry import kabsch, quaternion_to_matrix
from scipy.spatial.transform import Rotation


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_losses):
        super(MultiTaskLossWrapper, self).__init__()
        # Initialize learnable log variances for each loss
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


def rigid_from_3_points_batch(x1, x2, x3):
    """
    Compute the rigid transformation from 3 points using the Gram-Schmidt process for a batch of data.

    Parameters:
    - x1, x2, x3: Tensors of shape (batch_size, num_amino_acids, 3) representing the 3D coordinates of the points.

    Returns:
    - R: Rotation matrix of shape (batch_size, num_amino_acids, 3, 3).
    - t: Translation vector of shape (batch_size, num_amino_acids, 3).
    """
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / torch.norm(v1, dim=2, keepdim=True)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=2, keepdim=True)
    e2 = u2 / torch.norm(u2, dim=2, keepdim=True)
    e3 = torch.cross(e1, e2, dim=2)
    R = torch.stack([e1, e2, e3], dim=3)
    t = x2
    return R, t


def create_voxel_grid(coordinates, grid_size=32):
    # Determine the min and max bounds
    min_coords = np.min(coordinates, axis=0)
    max_coords = np.max(coordinates, axis=0)

    # Create an empty voxel grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # Scale and shift coordinates to fit within the voxel grid
    scale = (grid_size - 1) / (max_coords - min_coords)
    scaled_coords = (coordinates - min_coords) * scale

    # Assign points to the voxel grid
    for coord in scaled_coords:
        x, y, z = np.round(coord).astype(int)
        voxel_grid[x, y, z] = 1.0

    return voxel_grid


def calculate_surface_area(voxel_grid):
    # Create a convolution kernel to detect boundaries
    kernel = torch.tensor([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Apply the convolution
    voxel_grid = voxel_grid.unsqueeze(0).unsqueeze(0)
    boundary_voxels = F.conv3d(voxel_grid, kernel, padding=1).abs()

    # Count the number of boundary voxels
    surface_area = torch.sum(boundary_voxels)
    return surface_area


def surface_area_loss(predicted_coords, real_coords, grid_size=32):
    # Voxelize the predicted and real coordinates
    predicted_voxel_grid = create_voxel_grid(predicted_coords.detach().numpy(), grid_size)
    real_voxel_grid = create_voxel_grid(real_coords.detach().numpy(), grid_size)

    # Convert voxel grids to PyTorch tensors
    predicted_voxel_grid = torch.tensor(predicted_voxel_grid, dtype=torch.float32, requires_grad=True)
    real_voxel_grid = torch.tensor(real_voxel_grid, dtype=torch.float32)

    # Calculate the surface areas
    predicted_surface_area = calculate_surface_area(predicted_voxel_grid)
    real_surface_area = calculate_surface_area(real_voxel_grid)

    # Define the loss as the L2 difference between the surface areas
    loss = torch.nn.functional.mse_loss(predicted_surface_area, real_surface_area)
    return loss


def compute_distance_map(coordinates):
    """
    Calculate the pairwise distances between all atoms in the given coordinates.

    Args:
        coordinates (torch.Tensor): A tensor of shape (N, 3) where N is the number of atoms.

    Returns:
        torch.Tensor: A tensor of shape (N, N) containing the pairwise distances between atoms.
    """
    # Calculate the pairwise distances
    distance_map = torch.cdist(coordinates, coordinates, p=2)
    return distance_map


def distance_map_loss(predicted_coords, real_coords):
    """
    Compute the distance map loss between the predicted and real coordinates.

    Args:
        predicted_coords (torch.Tensor): A tensor of shape (N, 3) representing the predicted coordinates.
        real_coords (torch.Tensor): A tensor of shape (N, 3) representing the real coordinates.

    Returns:
        torch.Tensor: The computed distance map loss.
    """
    # Compute the distance maps
    predicted_distance_map = compute_distance_map(predicted_coords)
    real_distance_map = compute_distance_map(real_coords)

    # Define the loss as the L2 difference between the distance maps
    loss = torch.nn.functional.mse_loss(predicted_distance_map, real_distance_map)
    return loss


def calculate_radius_of_gyration(coordinates):
    # Calculate the centroid
    centroid = torch.mean(coordinates, dim=0)

    # Calculate squared distances from centroid
    squared_distances = torch.sum((coordinates - centroid) ** 2, dim=1)

    # Calculate radius of gyration
    radius_of_gyration = torch.sqrt(torch.mean(squared_distances))

    return radius_of_gyration


def radius_of_gyration_loss(predicted_coords, real_coords):
    # Calculate the radius of gyration for predicted and real coordinates
    predicted_radius_of_gyration = calculate_radius_of_gyration(predicted_coords)
    real_radius_of_gyration = calculate_radius_of_gyration(real_coords)

    # Define the loss as the L2 difference between the radii of gyration
    loss = torch.nn.functional.mse_loss(predicted_radius_of_gyration, real_radius_of_gyration)

    return loss


def compute_principal_components(coords):
    """
    Computes the principal components of the coordinates.

    Args:
        coords (torch.Tensor): The input coordinates of shape (batch, num_points, 3).

    Returns:
        torch.Tensor: The principal components.
    """
    # Center the coordinates
    centered_coords = coords - coords.mean(dim=0, keepdim=True)

    # Add a small amount of noise to avoid repeated singular values
    noise = torch.randn_like(centered_coords) * 1e-6
    centered_coords += noise

    # Compute SVD
    U, S, V = torch.svd(centered_coords)

    return V


def orientation_loss(pred_coords, true_coords):
    """
    Computes the orientation loss between predicted and true coordinates.

    Args:
        pred_coords (torch.Tensor): The predicted coordinates of shape (num_points, 3).
        true_coords (torch.Tensor): The true coordinates of shape (num_points, 3).

    Returns:
        torch.Tensor: The orientation loss.
    """
    try:
        pred_pc = compute_principal_components(pred_coords)
        true_pc = compute_principal_components(true_coords)
        # Compute the loss based on the principal components
        loss = torch.norm(pred_pc - true_pc, dim=-1).mean()
    except:
        loss = torch.tensor(0.0)

    return loss


def bond_lengths_loss(true_coords, pred_coords, masked_atoms):
    """
    Computes the bond lengths loss between true and predicted coordinates
    for the backbone atoms ['N', 'CA', 'C', 'O'] in a protein structure.

    Args:
        true_coords (torch.Tensor): Tensor of shape (batch, amino_acids, 4, 3)
                                    containing the true coordinates.
        pred_coords (torch.Tensor): Tensor of shape (batch, amino_acids, 4, 3)
                                    containing the predicted coordinates.
        masked_atoms (torch.Tensor): A boolean tensor of shape (batch, amino_acids)

    Returns:
        torch.Tensor: The computed bond lengths loss.
    """
    # Extract the relevant pairs of atoms for true coordinates
    n_ca_true_dist = torch.norm(true_coords[:, :, 0, :] - true_coords[:, :, 1, :], dim=2)
    ca_c_true_dist = torch.norm(true_coords[:, :, 1, :] - true_coords[:, :, 2, :], dim=2)
    c_o_true_dist = torch.norm(true_coords[:, :, 2, :] - true_coords[:, :, 3, :], dim=2)

    # Extract the relevant pairs of atoms for predicted coordinates
    n_ca_pred_dist = torch.norm(pred_coords[:, :, 0, :] - pred_coords[:, :, 1, :], dim=2)
    ca_c_pred_dist = torch.norm(pred_coords[:, :, 1, :] - pred_coords[:, :, 2, :], dim=2)
    c_o_pred_dist = torch.norm(pred_coords[:, :, 2, :] - pred_coords[:, :, 3, :], dim=2)

    # Compute the loss for each pair
    n_ca_loss = (n_ca_pred_dist - n_ca_true_dist) ** 2
    ca_c_loss = (ca_c_pred_dist - ca_c_true_dist) ** 2
    c_o_loss = (c_o_pred_dist - c_o_true_dist) ** 2

    n_ca_loss[masked_atoms == False] = 0
    ca_c_loss[masked_atoms == False] = 0
    c_o_loss[masked_atoms == False] = 0

    # Average the losses over all amino acids and the batch
    bond_length_loss = (n_ca_loss.mean() + ca_c_loss.mean() + c_o_loss.mean()) / 3

    return bond_length_loss


def bond_angles_loss(true_coords, pred_coords, masked_atoms):
    """
    Computes the bond angles loss between true and predicted coordinates
    for the backbone atoms ['N', 'CA', 'C', 'O'] in a protein structure.

    Args:
        true_coords (torch.Tensor): Tensor of shape (batch, amino_acids, 4, 3)
                                    containing the true coordinates.
        pred_coords (torch.Tensor): Tensor of shape (batch, amino_acids, 4, 3)
                                    containing the predicted coordinates.
        masked_atoms (torch.Tensor): A boolean tensor of shape (batch, amino_acids)

    Returns:
        torch.Tensor: The computed bond angles loss.
    """

    def compute_angle(a, b, c):
        # Vectors
        ba = a - b
        bc = c - b
        # Normalize vectors
        ba = ba / torch.norm(ba, dim=2, keepdim=True).clamp(min=1e-8)
        bc = bc / torch.norm(bc, dim=2, keepdim=True).clamp(min=1e-8)
        # Dot product and angle
        cos_angle = (ba * bc).sum(dim=2)
        angle = torch.acos(cos_angle.clamp(-1 + 1e-7, 1 - 1e-7))
        return angle

    # Compute angles for the true coordinates
    n_ca_true_angle = compute_angle(true_coords[:, :, 0, :], true_coords[:, :, 1, :], true_coords[:, :, 2, :])
    ca_c_true_angle = compute_angle(true_coords[:, :, 1, :], true_coords[:, :, 2, :], true_coords[:, :, 3, :])

    # Compute angles for the predicted coordinates
    n_ca_pred_angle = compute_angle(pred_coords[:, :, 0, :], pred_coords[:, :, 1, :], pred_coords[:, :, 2, :])
    ca_c_pred_angle = compute_angle(pred_coords[:, :, 1, :], pred_coords[:, :, 2, :], pred_coords[:, :, 3, :])

    # Compute the loss for each angle
    n_ca_loss = (n_ca_pred_angle - n_ca_true_angle) ** 2
    ca_c_loss = (ca_c_pred_angle - ca_c_true_angle) ** 2

    n_ca_loss[masked_atoms == False] = 0
    ca_c_loss[masked_atoms == False] = 0

    # Average the losses over all amino acids and the batch
    bond_angle_loss = (n_ca_loss.sum() + ca_c_loss.sum()) / masked_atoms.sum()

    return bond_angle_loss


def test_bond_angles_loss():
    """
    Test function for bond_angles_loss.
    Generates random predicted and true coordinates for testing.
    """
    # Example usage
    batch_size = 2  # Number of batches
    num_amino_acids = 10  # Number of amino acids per batch
    num_atoms = 4  # Number of atoms per amino acid
    num_coordinates = 3  # Number of coordinates (x, y, z)

    # Generate random predicted and true coordinates
    pred_coords = torch.randn(batch_size, num_amino_acids, num_atoms, num_coordinates)
    true_coords = torch.randn(batch_size, num_amino_acids, num_atoms, num_coordinates)
    masked_atoms = torch.randint(0, 2, (batch_size, num_amino_acids), dtype=torch.bool)

    # Calculate the bond angles loss
    loss = bond_angles_loss(true_coords, pred_coords, masked_atoms)

    print("Bond Angles Loss:", loss.item())


def test_bond_lengths_loss():
    """
    Test function for bond_lengths_loss.
    Generates random predicted and true coordinates for testing.
    """
    # Example usage
    batch_size = 2  # Number of batches
    num_amino_acids = 10  # Number of amino acids per batch
    num_atoms = 4  # Number of atoms per amino acid
    num_coordinates = 3  # Number of coordinates (x, y, z)

    # Generate random predicted and true coordinates
    pred_coords = torch.randn(batch_size, num_amino_acids, num_atoms, num_coordinates)
    true_coords = torch.randn(batch_size, num_amino_acids, num_atoms, num_coordinates)
    masked_atoms = torch.randint(0, 2, (batch_size, num_amino_acids), dtype=torch.bool)

    # Calculate the bond lengths loss
    loss = bond_lengths_loss(true_coords, pred_coords, masked_atoms)

    print("Bond Lengths Loss:", loss.item())


def test_orientation_loss():
    # Example usage
    N = 10  # Number of atoms
    pred_coords = torch.randn(N, 3)
    true_coords = torch.randn(N, 3)

    loss = orientation_loss(pred_coords, true_coords)
    print("Orientation Loss:", loss.item())


def test_distance_map_loss():
    # Example coordinates (replace with actual coordinates)
    predicted_coords = torch.randn(512, 3, requires_grad=True)
    real_coords = torch.randn(512, 3)

    # Calculate the loss
    loss = distance_map_loss(predicted_coords, real_coords)
    print("Distance Map Loss:", loss.item())


def test_surface_area_loss():
    # Example coordinates (replace with actual coordinates)
    predicted_coords = torch.randn(512, 3, requires_grad=True)
    real_coords = torch.randn(512, 3)

    # Calculate the loss
    loss = surface_area_loss(predicted_coords, real_coords)
    print("Surface Area Loss:", loss.item())


def test_radius_of_gyration_loss():
    # Example coordinates (replace with actual coordinates)
    predicted_coords = torch.randn(512, 3, requires_grad=True)
    real_coords = torch.randn(512, 3)

    # Calculate the loss
    loss = radius_of_gyration_loss(predicted_coords, real_coords)
    print("Radius of Gyration Loss:", loss.item())


def test_aligned_mse_loss():
    # Example usage
    batch_size = 2  # Number of batches
    seq_len = 10  # Sequence length
    num_atoms = 3  # Number of atoms per amino acid
    num_coordinates = 3  # Number of coordinates (x, y, z)

    # Generate random predicted and true coordinates
    x_true = torch.randn(batch_size, seq_len, num_atoms, num_coordinates)
    masks = torch.randint(0, 2, (batch_size, seq_len))

    # Randomly rotate and translate the true coordinates
    quat = torch.rand(batch_size, 4)
    rot = quaternion_to_matrix(quat)

    x_scale_factor = 4
    x_true_perturbed = rot.bmm(x_true.flatten(1, 2).mT).mT.reshape_as(x_true) + torch.rand(batch_size, 1, 1, 1) * x_scale_factor

    # Calculate the aligned MSE loss
    loss, x_true_perturbed_aligned, x_true_aligned = calculate_aligned_mse_loss(x_true_perturbed, x_true, masks, alignment_strategy='kbasch')
    rmsd = (x_true_perturbed_aligned - x_true_aligned).square().mean((-1, -2, -3)).sqrt()

    print(f"Aligned MSE Loss: {loss.mean().item()} (RMSD: {rmsd.mean().item()})")


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
        x_tru = x_true[i] # [seq_len, num_atoms, 3]

        with torch.no_grad():
            if alignment_strategy == 'kabsch':
                # Extract valid residues (each with multiple atoms) based on mask
                x_tru_valid = x_tru[mask]        # Shape: (num_valid_residues, num_atoms, 3)
                x_pred_valid = x_pred[mask]
                # Clone full true coords to fill in aligned residues
                x_true_aligned = x_tru.clone()
                # Flatten atoms across residues for Kabsch input
                coords_true_flat = x_tru_valid.reshape(-1, 3)
                coords_pred_flat = x_pred_valid.reshape(-1, 3)
                # Perform Kabsch alignment on flattened points
                aligned_flat = kabsch(coords_true_flat, coords_pred_flat,
                                      allow_reflections=True).detach()
                # Reshape back to per-residue atom layout
                aligned_valid = aligned_flat.reshape_as(x_tru_valid)
                # Assign aligned residues back into the full tensor
                x_true_aligned[mask] = aligned_valid

            elif alignment_strategy == 'kabsch_old':
                # Perform Kabsch alignment, keeping the same shape as the input
                x_true_aligned = kabsch_alignment(x_tru, x_pred, mask).detach()

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
        clamped_diff = torch.clamp(distance_diff, max=5**2)

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


def calculate_inverse_folding_loss(seq_logits, seq, masks):
    """
    Calculates the cross-entropy loss between seq_logits and seq after applying masks.

    Parameters:
    seq_logits (torch.Tensor): Logits of shape [batch_size, seq_len, num_classes].
    seq (torch.Tensor): True sequence of shape [batch_size, seq_len].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

    Returns:
    torch.Tensor: The computed cross-entropy loss for each batch element.
    """
    # Apply the mask to select only valid positions
    seq_masked = seq[masks]  # Shape: [num_valid]
    seq_logits_masked = seq_logits[masks]  # Shape: [num_valid, num_classes]

    # Compute cross-entropy loss using the masked areas
    loss = F.cross_entropy(seq_logits_masked, seq_masked, reduction='mean')
    return loss


@torch.no_grad()
def kabsch_alignment(x_true, x_predicted, mask):
    """
    Performs Kabsch alignment of x_true to x_predicted, with masked coordinates.

    Parameters:
    x_true (torch.Tensor): True coordinates of shape [seq_len, num_atoms, 3].
    x_predicted (torch.Tensor): Predicted coordinates of shape [seq_len, num_atoms, 3].
    mask (torch.Tensor): Boolean mask of shape [seq_len], where True indicates valid positions.

    Returns:
    torch.Tensor: The aligned x_true coordinates, with the same shape as the input.
    """
    # Apply the mask to select only valid positions
    x_tru_masked = x_true[mask]  # Shape: [num_valid, num_atoms, 3]
    x_pred_masked = x_predicted[mask]  # Shape: [num_valid, num_atoms, 3]

    # Reshape to [N, 3] where N = num_valid * num_atoms
    x_tru_flat = x_tru_masked.reshape(-1, 3)
    x_pred_flat = x_pred_masked.reshape(-1, 3)

    # Subtract centroids from masked coordinates
    centroid_true = x_tru_flat.mean(dim=0)
    centroid_pred = x_pred_flat.mean(dim=0)
    x_true_centered = x_tru_flat - centroid_true
    x_pred_centered = x_pred_flat - centroid_pred

    # Compute covariance matrix
    H = x_true_centered.T @ x_pred_centered

    # Singular Value Decomposition
    U, S, Vt = torch.linalg.svd(H)

    # Compute rotation matrix
    d = torch.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=x_true.device))
    R = Vt.T @ D @ U.T

    # Apply rotation to the masked true coordinates
    x_true_rotated = x_true_centered @ R

    # Translate back to the predicted centroid
    x_tru_aligned_flat = x_true_rotated + centroid_pred

    # Reshape aligned flat coordinates back to the original masked shape
    x_tru_aligned_masked = x_tru_aligned_flat.view_as(x_tru_masked)

    # Create a tensor to store the aligned true coordinates (same shape as input)
    x_tru_aligned = torch.zeros_like(x_true)

    # Place the aligned values back into their original positions
    x_tru_aligned[mask] = x_tru_aligned_masked

    return x_tru_aligned


def safe_normalize(tensor, dim=-1, keepdim=True, epsilon=1e-8):
    """
    Safely normalize a tensor along a dimension by adding epsilon to the norm before division
    for numerical stability.

    Args:
        tensor: Input tensor.
        dim: Dimension along which to normalize.
        keepdim: Whether the output tensor has dim retained or not.
        epsilon: Small value to prevent division by zero.

    Returns:
        Normalized tensor.
    """
    norm = torch.norm(tensor, dim=dim, keepdim=keepdim)
    # Add epsilon to the norm before division
    return tensor / (norm + epsilon)


def compute_local_frames(coords):
    """
    Compute local frames (rotation and translation) for each residue based on backbone atoms (N, CA, C),
    ensuring numerical stability.

    Follows a Gram-Schmidt-like process or definition similar to AlphaFold2:
    Origin (translation) at CA.
    x-axis points from CA to C.
    z-axis is orthogonal to the N-CA-C plane (using N-CA cross product with x-axis).
    y-axis completes the right-handed orthonormal basis.

    Args:
        coords: Tensor of shape [batch_size, seq_len, num_atoms, 3] or [seq_len, num_atoms, 3] 
               containing 3D coordinates. Assumes atom indices 0, 1, 2 correspond to N, CA, C respectively.

    Returns:
        rotation: Tensor of rotation matrices with same batch dimensions as input.
        translation: Tensor of translation vectors (CA positions) with same batch dimensions as input.
    """
    # Check if we have a batch dimension
    has_batch_dim = coords.dim() == 4
    
    # If no batch dimension, add one for consistent processing
    if not has_batch_dim:
        coords = coords.unsqueeze(0)  # Add batch dim of 1
    
    # Extract backbone atoms (N, CA, C)
    N = coords[:, :, 0, :]  # [batch_size, seq_len, 3]
    CA = coords[:, :, 1, :]  # [batch_size, seq_len, 3]
    C = coords[:, :, 2, :]  # [batch_size, seq_len, 3]

    # Define frame vectors relative to CA (origin)
    vec_ca_c = C - CA  # [batch_size, seq_len, 3]
    vec_n_ca = N - CA  # [batch_size, seq_len, 3]

    # Calculate x-axis (points from CA to C)
    # Explicitly maintain the dimensions for safe_normalize
    x_vec = safe_normalize(vec_ca_c, dim=-1, keepdim=True)  # [batch_size, seq_len, 3, 1]
    x_vec = x_vec.squeeze(-1)  # [batch_size, seq_len, 3]

    # Calculate z-axis (orthogonal to the N-CA-C plane)
    z_vec_unnormalized = torch.cross(vec_n_ca, x_vec, dim=-1)  # [batch_size, seq_len, 3]
    z_vec = safe_normalize(z_vec_unnormalized, dim=-1, keepdim=True)
    z_vec = z_vec.squeeze(-1)  # [batch_size, seq_len, 3]

    # Calculate y-axis (orthogonal to x and z to form a right-handed system)
    y_vec = torch.cross(z_vec, x_vec, dim=-1)  # [batch_size, seq_len, 3]
    # No need to normalize y_vec as x_vec and z_vec are already orthonormal
    # But we'll normalize anyway for numerical stability
    y_vec = safe_normalize(y_vec, dim=-1, keepdim=True)
    y_vec = y_vec.squeeze(-1)  # [batch_size, seq_len, 3]

    # Stack basis vectors as columns to form rotation matrices
    rotation = torch.stack([x_vec, y_vec, z_vec], dim=-1)  # Shape: [batch_size, seq_len, 3, 3]

    # Translation is the CA coordinate (origin of the frame)
    translation = CA  # Shape: [batch_size, seq_len, 3]
    
    # Remove batch dimension if input didn't have one
    if not has_batch_dim:
        rotation = rotation.squeeze(0)
        translation = translation.squeeze(0)

    return rotation, translation


def fape_loss_simplified(pred_coords, true_coords, clamp_distance=10.0, length_scale=10.0, epsilon=1e-8):
    """
    Compute the Simplified FAPE loss between predicted and true coordinates.

    *** IMPORTANT ***: This computes the FAPE loss comparing atoms within a residue only to the local frame of that SAME residue
    (i.e., only the i==j terms of the full FAPE calculation). It does NOT compute the full pairwise FAPE loss described in the
    AlphaFold 2 paper. It measures local frame rigidity.

    Args:
        pred_coords: Tensor of shape [batch_size, seq_len, num_atoms, 3] or [seq_len, num_atoms, 3], predicted coordinates.
        true_coords: Tensor of shape [batch_size, seq_len, num_atoms, 3] or [seq_len, num_atoms, 3], true coordinates.
        clamp_distance: Float, maximum distance error contribution per point (Angstroms).
        length_scale: Float, factor to scale the L2 distance error (often 10.0).
        epsilon: Small value for safe calculations.

    Returns:
        loss: Scalar tensor, the computed simplified FAPE loss.
    """
    # Check if inputs have batch dimension
    has_batch_dim = pred_coords.dim() == 4 and true_coords.dim() == 4
    
    # Add batch dimension if missing
    if not has_batch_dim:
        pred_coords = pred_coords.unsqueeze(0)
        true_coords = true_coords.unsqueeze(0)
        
    # Handle case where we might have fewer than 3 backbone atoms
    # Make sure we have at least N, CA, C atoms (indices 0, 1, 2)
    if pred_coords.shape[2] < 3 or true_coords.shape[2] < 3:
        # Return zero loss if we don't have enough atoms for frame computation
        return torch.tensor(0.0, device=pred_coords.device)
    
    # Compute local frames for true and predicted coordinates
    # Detach true_coords frames as they are targets, not to be optimized through
    try:
        with torch.no_grad():
            true_rotation, true_translation = compute_local_frames(true_coords)

        pred_rotation, pred_translation = compute_local_frames(pred_coords)

        # 1. Center coordinates at the frame origin (CA)
        true_coords_centered = true_coords - true_translation.unsqueeze(2)
        pred_coords_centered = pred_coords - pred_translation.unsqueeze(2)

        # 2. Apply inverse rotation (transpose of rotation matrix)
        true_rotation_inv = true_rotation.transpose(-2, -1)
        pred_rotation_inv = pred_rotation.transpose(-2, -1)

        # Transform true coordinates using true frame's inverse rotation
        true_coords_local = torch.einsum('blij,blmj->blmi', true_rotation_inv, true_coords_centered)

        # Transform predicted coordinates using predicted frame's inverse rotation
        pred_coords_local = torch.einsum('blij,blmj->blmi', pred_rotation_inv, pred_coords_centered)

        # Compute L2 distance error in the local frame
        distances = torch.sqrt(torch.sum((pred_coords_local - true_coords_local)**2, dim=-1) + epsilon)

        # Clamp distances
        clamped_distances = torch.clamp(distances, max=clamp_distance)

        # Compute the mean loss and scale it (divide by scale factor)
        loss = clamped_distances.mean() / length_scale
        
    except Exception as e:
        # If anything goes wrong (like degenerate frames), return a zero loss
        print(f"Warning: FAPE loss calculation failed with error: {e}")
        loss = torch.tensor(0.0, device=pred_coords.device)

    return loss


def calculate_fape_loss(x_predicted, x_true, masks, clamp_distance=10.0, length_scale=10.0):
    """
    Calculates the FAPE loss between x_predicted and x_true after applying masks.

    Parameters:
    x_predicted (torch.Tensor): Predicted coordinates of shape [batch_size, seq_len, num_atoms, 3].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.
    clamp_distance (float): Maximum distance error contribution per point (Angstroms).
    length_scale (float): Factor to scale the L2 distance error.

    Returns:
    torch.Tensor: The computed FAPE loss.
    """
    batch_size = x_predicted.shape[0]
    loss_list = []

    for i in range(batch_size):
        mask = masks[i].bool()  # Convert to boolean mask
        if (mask.sum() == 0):  # Skip if no valid residues
            loss_list.append(torch.tensor(0.0, device=x_predicted.device))
            continue
            
        # Extract only the valid residues
        x_pred = x_predicted[i][mask]  # [num_valid, num_atoms, 3]
        x_tru = x_true[i][mask]        # [num_valid, num_atoms, 3]
        
        # Ensure we have at least one residue
        if x_pred.shape[0] == 0:
            loss_list.append(torch.tensor(0.0, device=x_predicted.device))
            continue
            
        # Add batch dimension for the FAPE calculation
        x_pred = x_pred.unsqueeze(0)  # [1, num_valid, num_atoms, 3]
        x_tru = x_tru.unsqueeze(0)    # [1, num_valid, num_atoms, 3]
        
        try:
            loss = fape_loss_simplified(
                x_pred,
                x_tru,
                clamp_distance=clamp_distance,
                length_scale=length_scale
            )
            loss_list.append(loss)
        except Exception as e:
            # If anything goes wrong, add a zero loss for this batch item
            print(f"Warning: FAPE loss calculation failed for batch item {i} with error: {e}")
            loss_list.append(torch.tensor(0.0, device=x_predicted.device))

    return torch.stack(loss_list)


def calculate_decoder_loss(x_predicted, x_true, masks, configs, seq=None, dir_loss_logits=None, dist_loss_logits=None,
                           seq_logits=None, alignment_strategy='kabsch'):
    mse_loss, x_pred_aligned, x_true_aligned = calculate_aligned_mse_loss(x_predicted, x_true, masks,
                                                                          alignment_strategy=alignment_strategy)

    losses = []
    if configs.train_settings.losses.mse.enabled:
        losses.append(mse_loss.mean()*configs.train_settings.losses.mse.weight)

    if configs.train_settings.losses.backbone_distance.enabled:
        backbone_dist_loss = calculate_backbone_distance_loss(x_pred_aligned, x_true_aligned, masks).mean()
        losses.append(backbone_dist_loss*configs.train_settings.losses.backbone_distance.weight)

    if configs.train_settings.losses.backbone_direction.enabled:
        backbone_dir_loss = calculate_backbone_direction_loss(x_pred_aligned, x_true_aligned, masks).mean()
        losses.append(backbone_dir_loss*configs.train_settings.losses.backbone_direction.weight)

    if configs.train_settings.losses.binned_direction_classification.enabled:
        binned_dir_class_loss = calculate_binned_direction_classification_loss(dir_loss_logits, x_true_aligned, masks).mean() if dir_loss_logits is not None else 0.0
        losses.append(binned_dir_class_loss*configs.train_settings.losses.binned_direction_classification.weight)

    if configs.train_settings.losses.binned_distance_classification.enabled:
        binned_dist_class_loss = calculate_binned_distance_classification_loss(dist_loss_logits, x_true_aligned, masks).mean() if dist_loss_logits is not None else 0.0
        losses.append(binned_dist_class_loss*configs.train_settings.losses.binned_distance_classification.weight)

    if configs.train_settings.losses.inverse_folding.enabled:
        seq_loss = calculate_inverse_folding_loss(seq_logits, seq, masks.bool()) if seq_logits is not None else 0.0
        losses.append(seq_loss*configs.train_settings.losses.inverse_folding.weight)
        
    if configs.train_settings.losses.fape.enabled:
        fape_loss = calculate_fape_loss(
            x_predicted, 
            x_true, 
            masks,
            clamp_distance=configs.train_settings.losses.fape.clamp_distance,
            length_scale=configs.train_settings.losses.fape.length_scale
        ).mean()
        losses.append(fape_loss*configs.train_settings.losses.fape.weight)

    loss = sum(losses)

    return loss, x_pred_aligned, x_true_aligned


if __name__ == '__main__':
    test_surface_area_loss()
    test_distance_map_loss()
    test_radius_of_gyration_loss()
    test_orientation_loss()
    test_bond_lengths_loss()
    test_bond_angles_loss()
    test_aligned_mse_loss()

