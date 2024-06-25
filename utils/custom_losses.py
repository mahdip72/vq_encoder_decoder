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


def test_distance_map_loss():
    # Example coordinates (replace with actual coordinates)
    predicted_coords = torch.randn(512, 3, requires_grad=True)
    real_coords = torch.randn(512, 3)

    # Calculate the loss
    loss = distance_map_loss(predicted_coords, real_coords)
    print("Distance Map Loss:", loss.item())

