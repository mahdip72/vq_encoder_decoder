import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


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


def transform_predicted_points(R, t, x):
    """
    Transforms the predicted points using the batch of rotation matrices and translation vectors.

    Parameters:
    - R (torch.Tensor): Rotation matrices of shape (batch_size, num_amino_acids, 3, 3)
    - t (torch.Tensor): Translation vectors of shape (batch_size, num_amino_acids, 3)
    - x (torch.Tensor): Predicted points of shape (batch_size, num_amino_acids, 3)

    Returns:
    - torch.Tensor: Transformed points of shape (batch_size, num_amino_acids, 3)
    """

    # Calculate inverse rotation matrix
    R_inv = torch.linalg.inv(R)

    # Apply inverse rotation to t
    rotated_center = -torch.einsum('bnik,bnk->bni', R_inv, t)

    # Apply inverse transform to x
    x_transformed = torch.einsum('bnik,bnk->bni', R_inv, x)
    x_transformed += rotated_center

    return x_transformed


def fape_loss(x_predicted, x_true, Z=100.0, d_clamp=100.0, epsilon=1e-4):
    """
    Computes the Frame Aligned Point Error (FAPE) loss for a batch of predicted protein structures.

    Parameters:
    - x_true (torch.Tensor): True points of shape (batch_size, number_of_amino_acids, 3, 3)
    - x_predicted (torch.Tensor): Predicted points of shape (batch_size, number_of_amino_acids, 3, 3)
    - Z (float): Normalization factor
    - d_clamp (float): Clamping distance

    Returns:
    - float: Frame Aligned Point Error (FAPE)
    """
    batch_size, num_amino_acids, _, _ = x_true.shape

    # Compute the rigid transformation using the first three amino acids
    r_true, t_true = rigid_from_3_points_batch(x_true[:, :, 0, :],
                                               x_true[:, :, 1, :],
                                               x_true[:, :, 2, :])
    r_predicted, t_predicted = rigid_from_3_points_batch(x_predicted[:, :, 0, :],
                                                         x_predicted[:, :, 1, :],
                                                         x_predicted[:, :, 2, :])

    x_true_alpha_carbon = x_true[:, :, 1, :].squeeze(2)
    x_predicted_alpha_carbon = x_predicted[:, :, 1, :3].squeeze(2)

    # Transform all true and predicted points
    x_true_transformed = transform_predicted_points(r_true, t_true, x_true_alpha_carbon)
    x_predicted_transformed = transform_predicted_points(r_predicted, t_predicted, x_predicted_alpha_carbon)

    # Compute distances using L2 loss
    # distances = torch.nn.functional.mse_loss(x_true_transformed, x_predicted_transformed, reduction='none')
    # Compute distances using L2 norm manually
    distances = torch.sqrt(torch.sum((x_true_transformed - x_predicted_transformed) ** 2, dim=-1) + epsilon)

    # Clamp the distances by d_clamp
    clamped_distances = torch.minimum(distances, torch.tensor(d_clamp, dtype=distances.dtype))

    # Normalize by Z
    fape = clamped_distances / Z

    return fape


def get_axis_matrix(a, b, c, norm=True):
    """
    [This function is from the MP-NeRF project.]
    Gets an orthonormal basis as a matrix of [e1, e2, e3].
    Useful for constructing rotation matrices between planes
    according to the first answer here:
    https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
    Inputs:
    * a: (batch, 3) or (3, ). point(s) of the plane
    * b: (batch, 3) or (3, ). point(s) of the plane
    * c: (batch, 3) or (3, ). point(s) of the plane
    Outputs: orthonormal basis as a matrix of [e1, e2, e3]. calculated as:
        * e1_ = (c-b)
        * e2_proto = (b-a)
        * e3_ = e1_ ^ e2_proto
        * e2_ = e3_ ^ e1_
        * basis = normalize_by_vectors( [e1_, e2_, e3_] )
    Note: Could be done more by Graham-Schmidt and extend to N-dimensions
          but this is faster and more intuitive for 3D.
    """
    v1_ = c - b
    v2_ = b - a
    v3_ = torch.cross(v1_, v2_, dim=-1)
    v2_ready = torch.cross(v3_, v1_, dim=-1)
    basis    = torch.stack([v1_, v2_ready, v3_], dim=-2)
    # normalize if needed
    if norm:
        return basis / torch.norm(basis, dim=-1, keepdim=True)
    return basis


def fape_torch(pred_coords, true_coords, max_val=10., l_func=None,
               c_alpha=False, seq_list=None, rot_mats_g=None):
    """
    [This function is from the MP-NeRF project with some minor changes]
    Computes the Frame-Aligned Point Error. Scaled 0 <= FAPE <= 1
    Inputs:
    * pred_coords: (B, L, C, 3) predicted coordinates.
    * true_coords: (B, L, C, 3) ground truth coordinates.
    * max_val: maximum value (it's also the radius due to L1 usage)
    * l_func: function. allow for options other than l1 (consider dRMSD)
    * c_alpha: bool. whether to only calculate frames and loss from c_alphas
    * seq_list: list of strs (FASTA sequences). to calculate rigid bodies' indexs.
                Defaults to C-alpha if not passed.
    * rot_mats_g: optional. List of n_seqs x (N_frames, 3, 3) rotation matrices.

    Outputs: (B)
    """
    fape_store = []   # List of loss values for each sample

    if l_func is None:
        l_func = lambda x,y,eps=1e-7,sup=max_val: (((x-y)**2).sum(dim=-1) + eps).sqrt()

    # for chain
    for s in range(pred_coords.shape[0]):
        fape_store.append(0)
        cloud_mask = (torch.abs(true_coords[s]).sum(dim=-1) != 0)

        # center both structures
        pred_center = pred_coords[s] - pred_coords[s, cloud_mask].mean(dim=0, keepdim=True)
        true_center = true_coords[s] - true_coords[s, cloud_mask].mean(dim=0, keepdim=True)

        # Iterate through each residue
        num_residues = len(pred_center)
        for res_idx in range(num_residues):

            pred_center_res = pred_center[res_idx]
            true_center_res = true_center[res_idx]

            # get frames and conversions - same scheme as in mp_nerf proteins' concat of monomers
            if rot_mats_g is None:
                rigid_idxs = torch.tensor([0, 1, 2])  # Assume the backbone atoms are the first 3 atoms
                true_frames = get_axis_matrix(*true_center_res[rigid_idxs].detach(), norm=True)
                pred_frames = get_axis_matrix(*pred_center_res[rigid_idxs].detach(), norm=True)
                rot_mats = torch.matmul(torch.transpose(pred_frames, -1, -2), true_frames)
            else:
                rot_mats = rot_mats_g[s]

            # calculate loss only on c_alphas
            if c_alpha:
                mask_center = cloud_mask[res_idx]
                mask_center[:] = False
                mask_center[rigid_idxs[1]] = True

            fape = l_func(pred_center_res @ rot_mats, true_center_res).clamp(0, max_val)
            fape_store[s] += fape

        # Average the loss for current sample across all residues
        fape_store[s] /= num_residues

    # stack and average
    fape_store_tensor = torch.stack(fape_store, dim=0)

    # Final result is normalized to be between 0 and 1
    return (1/max_val) * fape_store_tensor


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
    loss = torch.nn.functional.mse_loss(predicted_distance_map, real_distance_map.squeeze(1))
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


def calculate_aligned_mse_loss(x_predicted, x_true, masks):
    """
    Calculates the MSE loss between x_predicted and x_true after performing Kabsch alignment,
    applying the provided masks.

    Parameters:
    x_predicted (torch.Tensor): Predicted coordinates of shape [batch_size, seq_len, num_atoms, 3].
    x_true (torch.Tensor): True coordinates of shape [batch_size, seq_len, num_atoms, 3].
    masks (torch.Tensor): Binary masks of shape [batch_size, seq_len], where 1 indicates valid positions.

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
        x_tru = x_true[i]

        # Perform Kabsch alignment, keeping the same shape as the input
        x_true_aligned = kabsch_alignment(x_tru, x_pred, mask).detach()
        x_true_aligned_list.append(x_true_aligned)

        # Compute MSE loss using the masked areas
        loss = torch.mean(((x_pred[mask] - x_true_aligned[mask]) ** 2))
        loss_list.append(loss)

    return torch.stack(loss_list), x_predicted, torch.stack(x_true_aligned_list)


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


def calculate_decoder_loss(x_predicted, x_true, masks, configs, seq=None, dir_loss_logits=None, dist_loss_logits=None, seq_logits=None):
    kabsch_loss, x_pred_aligned, x_true_aligned = calculate_aligned_mse_loss(x_predicted, x_true, masks)

    losses = []
    if configs.train_settings.losses.kabsch.enable:
        losses.append(kabsch_loss.mean()*configs.train_settings.losses.kabsch.weight)
    if configs.train_settings.losses.backbone_distance.enable:
        backbone_dist_loss = calculate_backbone_distance_loss(x_predicted, x_true, masks).mean()
        losses.append(backbone_dist_loss*configs.train_settings.losses.backbone_distance.weight)

    if configs.train_settings.losses.backbone_direction.enable:
        backbone_dir_loss = calculate_backbone_direction_loss(x_predicted, x_true, masks).mean()
        losses.append(backbone_dir_loss*configs.train_settings.losses.backbone_direction.weight)

    if configs.train_settings.losses.binned_direction_classification.enable:
        binned_dir_class_loss = calculate_binned_direction_classification_loss(dir_loss_logits, x_true, masks).mean() if dir_loss_logits is not None else 0.0
        losses.append(binned_dir_class_loss*configs.train_settings.losses.binned_direction_classification.weight)

    if configs.train_settings.losses.binned_distance_classification.enable:
        binned_dist_class_loss = calculate_binned_distance_classification_loss(dist_loss_logits, x_true, masks).mean() if dist_loss_logits is not None else 0.0
        losses.append(binned_dist_class_loss*configs.train_settings.losses.binned_distance_classification.weight)

    if configs.train_settings.losses.inverse_folding.enable:
        seq_loss = calculate_inverse_folding_loss(seq_logits, seq, masks.bool()) if seq_logits is not None else 0.0
        losses.append(seq_loss*configs.train_settings.losses.inverse_folding.weight)

    loss = sum(losses)

    return loss, x_pred_aligned, x_true_aligned


if __name__ == '__main__':
    test_surface_area_loss()
    test_distance_map_loss()
    test_radius_of_gyration_loss()
    test_orientation_loss()
    test_bond_lengths_loss()
    test_bond_angles_loss()
