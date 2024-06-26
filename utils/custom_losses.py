import numpy as np
import torch
import torch.nn.functional as F


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
    # Calculate the pairwise distances
    distance_map = torch.cdist(coordinates, coordinates, p=2)
    return distance_map


def distance_map_loss(predicted_coords, real_coords):
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
    Compute the principal components of a set of coordinates.

    Args:
        coords (torch.Tensor): Tensor of shape (N, 3) where N is the number of atoms.

    Returns:
        torch.Tensor: Principal components of shape (3, 3).
    """
    # Subtract the mean to center the coordinates
    centered_coords = coords - coords.mean(dim=0)

    # Perform Singular Value Decomposition (SVD)
    _, _, V = torch.svd(centered_coords)

    return V


def orientation_loss(pred_coords, true_coords):
    """
    Compute the orientation loss based on the principal components.

    Args:
        pred_coords (torch.Tensor): Predicted coordinates of shape (N, 3).
        true_coords (torch.Tensor): True coordinates of shape (N, 3).

    Returns:
        torch.Tensor: The orientation loss.
    """
    # Compute the principal components of the predicted and true coordinates
    pred_pc = compute_principal_components(pred_coords)
    true_pc = compute_principal_components(true_coords)

    # Compute the alignment error between the principal components
    alignment_error = torch.sum(1 - torch.abs(torch.sum(pred_pc * true_pc, dim=0)))

    return alignment_error


def bond_lengths_loss(true_coords, pred_coords):
    """
    Computes the bond lengths loss between true and predicted coordinates
    for the backbone atoms ['N', 'CA', 'C', 'O'] in a protein structure.

    Args:
        true_coords (torch.Tensor): Tensor of shape (batch, amino_acids, 4, 3)
                                    containing the true coordinates.
        pred_coords (torch.Tensor): Tensor of shape (batch, amino_acids, 4, 3)
                                    containing the predicted coordinates.

    Returns:
        torch.Tensor: The computed bond lengths loss.
    """
    # Define the ideal bond lengths (in Ångströms)
    ideal_lengths = {
        ('N', 'CA'): 1.46,
        ('CA', 'C'): 1.53,
        ('C', 'O'): 1.23
    }

    # Extract the relevant pairs of atoms
    n_ca_dist = torch.norm(pred_coords[:, :, 0, :] - pred_coords[:, :, 1, :], dim=2)
    ca_c_dist = torch.norm(pred_coords[:, :, 1, :] - pred_coords[:, :, 2, :], dim=2)
    c_o_dist = torch.norm(pred_coords[:, :, 2, :] - pred_coords[:, :, 3, :], dim=2)

    # Compute the loss for each pair
    n_ca_loss = (n_ca_dist - ideal_lengths[('N', 'CA')]) ** 2
    ca_c_loss = (ca_c_dist - ideal_lengths[('CA', 'C')]) ** 2
    c_o_loss = (c_o_dist - ideal_lengths[('C', 'O')]) ** 2

    # Average the losses over all amino acids and the batch
    bond_length_loss = (n_ca_loss.mean() + ca_c_loss.mean() + c_o_loss.mean()) / 3

    return bond_length_loss


def bond_angles_loss(true_coords, pred_coords):
    """
    Computes the bond angles loss between true and predicted coordinates
    for the backbone atoms ['N', 'CA', 'C', 'O'] in a protein structure.

    Args:
        true_coords (torch.Tensor): Tensor of shape (batch, amino_acids, 4, 3)
                                    containing the true coordinates.
        pred_coords (torch.Tensor): Tensor of shape (batch, amino_acids, 4, 3)
                                    containing the predicted coordinates.

    Returns:
        torch.Tensor: The computed bond angles loss.
    """
    # Ideal bond angles (in degrees)
    ideal_angles = {
        ('N', 'CA', 'C'): 110.6,
        ('CA', 'C', 'O'): 120.0
    }

    # Convert angles to radians
    ideal_angles = {k: torch.tensor(v * (torch.pi / 180.0)) for k, v in ideal_angles.items()}

    def compute_angle(a, b, c):
        # Vectors
        ba = a - b
        bc = c - b
        # Normalize vectors
        ba = ba / torch.norm(ba, dim=2, keepdim=True)
        bc = bc / torch.norm(bc, dim=2, keepdim=True)
        # Dot product and angle
        cos_angle = (ba * bc).sum(dim=2)
        angle = torch.acos(cos_angle)
        return angle

    # Compute angles for the predicted and true coordinates
    n_ca_c_pred = compute_angle(pred_coords[:, :, 0, :], pred_coords[:, :, 1, :], pred_coords[:, :, 2, :])
    ca_c_o_pred = compute_angle(pred_coords[:, :, 1, :], pred_coords[:, :, 2, :], pred_coords[:, :, 3, :])

    n_ca_c_true = compute_angle(true_coords[:, :, 0, :], true_coords[:, :, 1, :], true_coords[:, :, 2, :])
    ca_c_o_true = compute_angle(true_coords[:, :, 1, :], true_coords[:, :, 2, :], true_coords[:, :, 3, :])

    # Compute the loss for each angle
    n_ca_c_loss = (n_ca_c_pred - ideal_angles[('N', 'CA', 'C')]) ** 2
    ca_c_o_loss = (ca_c_o_pred - ideal_angles[('CA', 'C', 'O')]) ** 2

    # Average the losses over all amino acids and the batch
    bond_angle_loss = (n_ca_c_loss.mean() + ca_c_o_loss.mean()) / 2

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

    # Calculate the bond angles loss
    loss = bond_angles_loss(true_coords, pred_coords)

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

    # Calculate the bond lengths loss
    loss = bond_lengths_loss(true_coords, pred_coords)

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
