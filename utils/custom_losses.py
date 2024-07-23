import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def rigidFrom3Points(x1, x2, x3):
    """
    Compute the rigid transformation from 3 points using the Gram-Schmidt process.

    Parameters:
    - x1, x2, x3: Tensors of shape (B, 3) representing the 3D coordinates of the points for a batch size B.

    Returns:
    - R: Rotation matrix of shape (B, 3, 3).
    - t: Translation vector of shape (B, 3).
    """
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = u2 / torch.norm(u2, dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2)
    R = torch.stack([e1, e2, e3], dim=-1)
    t = x2
    return R, t


def compute_fape(T_pred, x_pred, T_true, x_true, Z=10.0, d_clamp=10.0, epsilon=1e-4):
    """
    Compute the Frame Aligned Point Error (FAPE) loss for a batch of data.

    The FAPE loss measures the alignment error between a set of predicted points and true points
    after applying the respective predicted and true transformations. It is used for tasks such as
    evaluating the accuracy of protein structure predictions and aligning molecular structures.

    The algorithm involves the following steps:
    1. Transform the predicted points into the true frame using the predicted transformations.
    2. Transform the true points into the predicted frame using the inverse of the true transformations.
    3. Calculate the Euclidean distances between the transformed predicted and true points.
    4. Clamp the distances to avoid large errors dominating the loss.
    5. Compute the mean of the clamped distances and normalize by a factor Z.

    Parameters:
    - T_pred: Tensor of shape (B, N_frames, 3, 4) representing predicted transformations.
              Each transformation consists of a 3x3 rotation matrix and a 3D translation vector.
    - x_pred: Tensor of shape (B, N_frames, N_atoms, 3) representing predicted points.
    - T_true: Tensor of shape (B, N_frames, 3, 4) representing true transformations.
              Each transformation consists of a 3x3 rotation matrix and a 3D translation vector.
    - x_true: Tensor of shape (B, N_frames, N_atoms, 3) representing true points.
    - Z: Normalization factor to scale the loss.
    - d_clamp: Clamping distance to avoid large errors dominating the loss.
    - epsilon: Small value to prevent numerical instability during distance calculation.

    Returns:
    - FAPE loss: A scalar value representing the frame aligned point error for the batch.
    """

    # Number of frames and atoms
    B = T_pred.shape[0]
    N_frames = T_pred.shape[1]
    N_atoms = x_pred.shape[2]

    # Transform predicted points to the true frame
    x_pred_trans = torch.einsum('bfij,bfkj->bfki', T_pred[:, :, :3], x_pred) + T_pred[:, :, 3].unsqueeze(-2)

    # Transform true points to the predicted frame
    T_true_inv = torch.linalg.inv(torch.cat([T_true[:, :, :3], T_true[:, :, 3:]], dim=-1))
    x_true_trans = torch.einsum('bfij,bfkj->bfki', T_true_inv[:, :, :3], x_true) + T_true_inv[:, :, 3].unsqueeze(-2)

    # Calculate squared distances with epsilon
    d_ij = torch.sqrt(torch.sum((x_pred_trans - x_true_trans) ** 2, dim=-1) + epsilon)

    # Clamp distances
    d_ij_clamped = torch.minimum(d_ij, torch.tensor(d_clamp, device=d_ij.device))

    # Calculate FAPE loss
    L_FAPE = (1 / Z) * torch.mean(d_ij_clamped)

    return L_FAPE


def test_fape_loss_batch():
    # I am not sure about the correctness of this test, but it is a good starting point
    # Define multiple sets of coordinates for a batch
    coords_pred = [
        [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])],
        [torch.tensor([2.0, 0.0, 0.0]), torch.tensor([0.0, 2.0, 0.0]), torch.tensor([0.0, 0.0, 2.0])],
        [torch.tensor([1.5, 0.5, 0.5]), torch.tensor([0.5, 1.5, 0.5]), torch.tensor([0.5, 0.5, 1.5])]
    ]
    coords_true = [
        [torch.tensor([1.1, 0.1, 0.1]), torch.tensor([0.1, 1.1, 0.1]), torch.tensor([0.1, 0.1, 1.1])],
        [torch.tensor([2.1, 0.1, 0.1]), torch.tensor([0.1, 2.1, 0.1]), torch.tensor([0.1, 0.1, 2.1])],
        [torch.tensor([1.6, 0.6, 0.6]), torch.tensor([0.6, 1.6, 0.6]), torch.tensor([0.6, 0.6, 1.6])]
    ]

    B = len(coords_pred)  # Batch size
    N_frames = 1
    N_atoms = 3

    # Prepare tensors
    x_pred = torch.stack([torch.stack(batch) for batch in coords_pred]).unsqueeze(1)  # Shape: (B, N_frames, N_atoms, 3)
    x_true = torch.stack([torch.stack(batch) for batch in coords_true]).unsqueeze(1)  # Shape: (B, N_frames, N_atoms, 3)

    T_pred = []
    T_true = []

    for batch_pred, batch_true in zip(coords_pred, coords_true):
        R_pred, t_pred = rigidFrom3Points(*batch_pred)
        R_true, t_true = rigidFrom3Points(*batch_true)

        T_pred_batch = torch.eye(4).repeat(N_frames, 1, 1)
        T_true_batch = torch.eye(4).repeat(N_frames, 1, 1)

        T_pred_batch[:, :3, :3] = R_pred
        T_pred_batch[:, :3, 3] = t_pred

        T_true_batch[:, :3, :3] = R_true
        T_true_batch[:, :3, 3] = t_true

        T_pred.append(T_pred_batch)
        T_true.append(T_true_batch)

    T_pred = torch.stack(T_pred)  # Shape: (B, N_frames, 4, 4)
    T_true = torch.stack(T_true)  # Shape: (B, N_frames, 4, 4)

    # Compute FAPE loss
    loss = compute_fape(T_pred, x_pred, T_true, x_true)

    return loss


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


def distance_map_loss(predicted_coords, real_distance_map):
    """
    Compute the distance map loss between the predicted and real coordinates.

    Args:
        predicted_coords (torch.Tensor): A tensor of shape (N, 3) representing the predicted coordinates.
        real_distance_map (torch.Tensor): A tensor of shape (N, N) representing the real distance map.

    Returns:
        torch.Tensor: The computed distance map loss.
    """
    # Compute the distance maps
    predicted_distance_map = compute_distance_map(predicted_coords)
    # real_distance_map = compute_distance_map(real_coords)

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


if __name__ == '__main__':
    test_surface_area_loss()
    test_distance_map_loss()
    test_radius_of_gyration_loss()
    test_orientation_loss()
    test_bond_lengths_loss()
    test_bond_angles_loss()
