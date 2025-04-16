import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.spatial.transform import Rotation

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.custom_losses import fape_loss_simplified, calculate_fape_loss
from utils.custom_losses import quaternion_to_matrix


def generate_protein_like_structure(seq_len=10, num_atoms=3, device='cpu'):
    """
    Generate a synthetic protein-like structure with realistic backbone geometry.
    
    Args:
        seq_len: Number of amino acids
        num_atoms: Number of atoms per residue (at least 3 for N, CA, C)
        device: Device to store tensors on
        
    Returns:
        coords: Tensor of shape [seq_len, num_atoms, 3]
    """
    # Create an extended chain backbone structure
    coords = torch.zeros(seq_len, num_atoms, 3, device=device)
    
    # Define typical distances
    ca_c_dist = 1.52  # Å
    c_n_dist = 1.33   # Å
    n_ca_dist = 1.47  # Å
    
    # Create coordinates with realistic backbone geometry
    for i in range(seq_len):
        # Position the CA atom
        coords[i, 1, 0] = i * 3.8  # Place CA atoms ~3.8Å apart along x-axis
        
        # Position the C atom
        coords[i, 2, 0] = coords[i, 1, 0] + ca_c_dist * 0.5
        coords[i, 2, 1] = ca_c_dist * 0.866
        
        # Position the N atom
        if i > 0:
            # Connect to previous residue
            coords[i, 0, 0] = coords[i, 1, 0] - n_ca_dist * 0.5
            coords[i, 0, 1] = n_ca_dist * 0.866
        else:
            # First residue N position
            coords[i, 0, 0] = coords[i, 1, 0] - n_ca_dist * 0.5
            coords[i, 0, 1] = -n_ca_dist * 0.866
    
    # Add some random variation to make it more realistic
    noise = torch.randn_like(coords) * 0.1
    coords = coords + noise
    
    return coords


def apply_random_transformation(coords, rotation_scale=1.0, translation_scale=10.0):
    """
    Apply a random rigid transformation (rotation + translation) to coordinates.
    
    Args:
        coords: Tensor of shape [..., 3]
        rotation_scale: Scale factor for rotation (0 to 1, where 1 is full random rotation)
        translation_scale: Scale factor for translation
        
    Returns:
        transformed_coords: Transformed coordinates with same shape as input
    """
    device = coords.device
    original_shape = coords.shape
    
    # Reshape to [N, 3] for transformation
    flat_coords = coords.reshape(-1, 3)
    
    # Generate random rotation matrix
    if rotation_scale > 0:
        random_rotation = Rotation.random()
        # Scale the rotation (interpolate between identity and random rotation)
        if rotation_scale < 1.0:
            identity = np.eye(3)
            rotation_matrix = (1 - rotation_scale) * identity + rotation_scale * random_rotation.as_matrix()
            # Re-normalize to ensure it's a valid rotation
            rotation = Rotation.from_matrix(rotation_matrix)
            rotation_matrix = rotation.as_matrix()
        else:
            rotation_matrix = random_rotation.as_matrix()
    else:
        rotation_matrix = np.eye(3)
    
    rotation_tensor = torch.tensor(rotation_matrix, dtype=torch.float32, device=device)
    
    # Generate random translation
    translation = torch.randn(3, device=device) * translation_scale
    
    # Apply rotation
    rotated = torch.matmul(flat_coords, rotation_tensor.T)
    
    # Apply translation
    transformed = rotated + translation
    
    # Reshape back to original shape
    transformed_coords = transformed.reshape(original_shape)
    
    return transformed_coords, rotation_tensor, translation


def test_rotation_invariance(batch_size=2, seq_len=10, num_atoms=4, num_tests=10, 
                             rotation_scales=[0, 0.5, 1.0], 
                             translation_scales=[0, 5.0, 10.0],
                             device='cpu'):
    """
    Test if FAPE loss is invariant to global rotations and translations.
    
    Args:
        batch_size: Number of protein structures in batch
        seq_len: Number of amino acids per protein
        num_atoms: Number of atoms per residue
        num_tests: Number of random tests to run
        rotation_scales: List of rotation scales to test
        translation_scales: List of translation scales to test
        device: Device to run tests on
        
    Returns:
        results: Dictionary of test results
    """
    results = {
        'rotation_only': [],
        'translation_only': [],
        'rotation_and_translation': [],
        'deformation': []  # Tests where the structure is actually deformed
    }
    
    for test_idx in range(num_tests):
        print(f"Running test {test_idx + 1}/{num_tests}")
        
        # Generate random protein-like structures
        coords_batch = torch.stack([generate_protein_like_structure(seq_len, num_atoms, device) 
                                   for _ in range(batch_size)])
        
        # Create masks (all valid in this test)
        masks = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        
        # Test rotation invariance at different scales
        for rot_scale in rotation_scales:
            for trans_scale in translation_scales:
                # Create a batch of transformed coordinates
                transformed_batch = []
                for b in range(batch_size):
                    transformed, _, _ = apply_random_transformation(
                        coords_batch[b], 
                        rotation_scale=rot_scale,
                        translation_scale=trans_scale
                    )
                    transformed_batch.append(transformed)
                
                transformed_batch = torch.stack(transformed_batch)
                
                # Calculate FAPE loss
                fape_loss = calculate_fape_loss(
                    transformed_batch, 
                    coords_batch, 
                    masks,
                    clamp_distance=10.0,
                    length_scale=10.0
                ).mean().item()
                
                # Store results based on transformation type
                if rot_scale > 0 and trans_scale > 0:
                    results['rotation_and_translation'].append(fape_loss)
                elif rot_scale > 0:
                    results['rotation_only'].append(fape_loss)
                elif trans_scale > 0:
                    results['translation_only'].append(fape_loss)
                else:
                    # Identity transformation, loss should be exactly 0
                    assert fape_loss < 1e-5, f"Identity transform should give zero loss, got {fape_loss}"
        
        # Test with actual deformation (not just rigid transformation)
        deformed = coords_batch + torch.randn_like(coords_batch) * 0.5
        deform_loss = calculate_fape_loss(
            deformed, 
            coords_batch, 
            masks,
            clamp_distance=10.0,
            length_scale=10.0
        ).mean().item()
        results['deformation'].append(deform_loss)
    
    # Calculate statistics
    for key in results:
        if results[key]:
            mean_val = np.mean(results[key])
            std_val = np.std(results[key])
            results[key] = {
                'values': results[key],
                'mean': mean_val,
                'std': std_val
            }
    
    return results


def test_extreme_transformations(seq_len=10, num_atoms=4, device='cpu'):
    """
    Test FAPE loss on extreme transformations to verify its invariance properties.
    
    Args:
        seq_len: Number of amino acids
        num_atoms: Number of atoms per residue
        device: Device to run tests on
        
    Returns:
        results: Dictionary of test results
    """
    # Generate a single protein-like structure
    coords = generate_protein_like_structure(seq_len, num_atoms, device).unsqueeze(0)
    masks = torch.ones(1, seq_len, device=device, dtype=torch.bool)
    
    # Get the data type of the coordinates for consistent tensor creation
    coords_dtype = coords.dtype
    
    # List of transformations to test
    transforms = [
        ("No transformation", lambda x: x.clone()),
        ("180° rotation around X", lambda x: torch.matmul(x.reshape(-1, 3), 
                                                       torch.tensor([[1, 0, 0], 
                                                                    [0, -1, 0], 
                                                                    [0, 0, -1]], 
                                                                   device=device,
                                                                   dtype=coords_dtype)).reshape_as(x)),
        ("90° rotation around Y", lambda x: torch.matmul(x.reshape(-1, 3), 
                                                      torch.tensor([[0, 0, 1], 
                                                                   [0, 1, 0], 
                                                                   [-1, 0, 0]], 
                                                                  device=device,
                                                                  dtype=coords_dtype)).reshape_as(x)),
        ("Large translation", lambda x: x + torch.tensor([100.0, -50.0, 200.0], 
                                                      device=device, 
                                                      dtype=coords_dtype))
    ]
    
    results = {}
    
    for name, transform_fn in transforms:
        # Apply transformation
        transformed = transform_fn(coords)
        
        # Calculate FAPE loss
        fape_loss = calculate_fape_loss(transformed, coords, masks).item()
        
        results[name] = fape_loss
        print(f"{name}: FAPE Loss = {fape_loss}")
    
    return results


def plot_results(results):
    """Plot the FAPE loss results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    labels = ['Rotation Only', 'Translation Only', 'Rotation & Translation', 'Deformation']
    data = [results['rotation_only']['mean'], results['translation_only']['mean'], 
            results['rotation_and_translation']['mean'], results['deformation']['mean']]
    std_devs = [results['rotation_only']['std'], results['translation_only']['std'], 
                results['rotation_and_translation']['std'], results['deformation']['std']]
    
    # Create the bar chart
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, data, yerr=std_devs, align='center', alpha=0.7, capsize=10)
    
    # Set labels and title
    ax.set_ylabel('FAPE Loss')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('FAPE Loss Under Different Transformations')
    ax.yaxis.grid(True)
    
    # Red dashed line at y=0 for reference
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Formatting
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_dir / 'fape_loss_verification.png'))
    
    plt.show()


def print_results(results):
    """Print the results in a formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Transformation Type':<25} | {'Mean FAPE Loss':<15} | {'Std Dev':<15}")
    print("-" * 80)
    
    for transform_type in ['rotation_only', 'translation_only', 'rotation_and_translation', 'deformation']:
        name = transform_type.replace('_', ' ').title()
        mean_val = results[transform_type]['mean']
        std_val = results[transform_type]['std']
        print(f"{name:<25} | {mean_val:<15.6f} | {std_val:<15.6f}")
    
    print("=" * 80 + "\n")
    
    print("Interpretation:")
    for transform_type in ['rotation_only', 'translation_only', 'rotation_and_translation']:
        if results[transform_type]['mean'] < 1e-4:
            print(f"✓ {transform_type.replace('_', ' ').title()}: FAPE loss is effectively zero (invariant)")
        elif results[transform_type]['mean'] < 1e-2:
            print(f"? {transform_type.replace('_', ' ').title()}: FAPE loss is very small, likely numerical precision issues")
        else:
            print(f"✗ {transform_type.replace('_', ' '). title()}: FAPE loss is non-zero, the implementation may have issues")
    
    if results['deformation']['mean'] > 0.01:
        print(f"✓ Deformation: FAPE loss correctly detects structural changes")
    else:
        print(f"✗ Deformation: FAPE loss failed to detect structural changes")


if __name__ == "__main__":
    print("Testing FAPE loss invariance to rigid transformations...")
    
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on {device}")
    
    # Run the invariance tests
    results = test_rotation_invariance(
        batch_size=4,
        seq_len=20,
        num_atoms=4,
        num_tests=5,
        rotation_scales=[0, 0.5, 1.0],
        translation_scales=[0, 5.0, 10.0],
        device=device
    )
    
    print_results(results)
    plot_results(results)
    
    print("\nTesting FAPE loss with extreme transformations...")
    extreme_results = test_extreme_transformations(
        seq_len=20,
        num_atoms=4,
        device=device
    )
    
    print("\nComplete! If all tests pass, the FAPE loss implementation is correctly invariant to rigid transformations.")

