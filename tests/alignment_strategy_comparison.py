import sys
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.custom_losses import calculate_aligned_mse_loss, quaternion_to_matrix

def compare_alignment_strategies(batch_size=2, seq_len=10, num_atoms=3, perturbation_scale=4, 
                                runs=5, strategies=None, device='cpu'):
    """
    Compare different alignment strategies based on MSE and RMSD metrics.
    
    Args:
        batch_size: Number of batches in the tensor
        seq_len: Sequence length
        num_atoms: Number of atoms per amino acid
        perturbation_scale: Scale factor for perturbation
        runs: Number of runs to average results over
        strategies: List of alignment strategies to test
        device: Device to run the test on ('cpu' or 'cuda')
        
    Returns:
        results: Dictionary containing the results
    """
    if strategies is None:
        strategies = ['kabsch', 'kabsch_old', 'quaternion']
        
    results = {strategy: {'mse': [], 'rmsd': [], 'time': []} for strategy in strategies}
    
    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        
        # Generate random true coordinates
        x_true = torch.randn(batch_size, seq_len, num_atoms, 3, device=device)
        
        # Generate random masks (1 for valid positions, 0 for invalid)
        masks = torch.randint(0, 2, (batch_size, seq_len), device=device)
        
        # Ensure at least one position is valid in each batch
        for b in range(batch_size):
            if masks[b].sum() == 0:
                masks[b, 0] = 1
        
        # Randomly rotate and translate the true coordinates
        quat = torch.rand(batch_size, 4, device=device)
        rot = quaternion_to_matrix(quat)
        
        # Apply rotation and translation to create perturbed coordinates
        x_true_perturbed = rot.bmm(x_true.flatten(1, 2).mT).mT.reshape_as(x_true) + \
                           torch.rand(batch_size, 1, 1, 1, device=device) * perturbation_scale
        
        # Test each alignment strategy
        for strategy in strategies:
            start_time = time.time()
            loss, aligned_pred, aligned_true = calculate_aligned_mse_loss(
                x_true_perturbed, x_true, masks, alignment_strategy=strategy
            )
            end_time = time.time()
            
            # Calculate RMSD
            rmsd = (aligned_pred - aligned_true).square().mean((-1, -2, -3)).sqrt()
            
            # Store results
            results[strategy]['mse'].append(loss.mean().item())
            results[strategy]['rmsd'].append(rmsd.mean().item())
            results[strategy]['time'].append(end_time - start_time)
    
    # Average the results
    for strategy in strategies:
        for metric in ['mse', 'rmsd', 'time']:
            results[strategy][metric] = {
                'mean': np.mean(results[strategy][metric]),
                'std': np.std(results[strategy][metric]),
                'values': results[strategy][metric]
            }
    
    return results

def print_results(results):
    """Print the results in a formatted table."""
    print("\n" + "="*80)
    print(f"{'Strategy':<15} | {'MSE Mean':<12} | {'MSE StdDev':<12} | {'RMSD Mean':<12} | {'RMSD StdDev':<12} | {'Time (s)':<12}")
    print("-"*80)
    
    for strategy, metrics in results.items():
        print(f"{strategy:<15} | "
              f"{metrics['mse']['mean']:<12.6f} | "
              f"{metrics['mse']['std']:<12.6f} | "
              f"{metrics['rmsd']['mean']:<12.6f} | "
              f"{metrics['rmsd']['std']:<12.6f} | "
              f"{metrics['time']['mean']:<12.6f}")
    
    print("="*80 + "\n")

def plot_results(results, save_path=None):
    """Plot the results as a bar chart."""
    strategies = list(results.keys())
    
    # Extract data for plotting
    mse_means = [results[s]['mse']['mean'] for s in strategies]
    mse_stds = [results[s]['mse']['std'] for s in strategies]
    rmsd_means = [results[s]['rmsd']['mean'] for s in strategies]
    rmsd_stds = [results[s]['rmsd']['std'] for s in strategies]
    time_means = [results[s]['time']['mean'] for s in strategies]
    time_stds = [results[s]['time']['std'] for s in strategies]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot MSE
    ax1.bar(strategies, mse_means, yerr=mse_stds, capsize=5, alpha=0.7)
    ax1.set_title('MSE Comparison')
    ax1.set_ylabel('Mean Squared Error')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot RMSD
    ax2.bar(strategies, rmsd_means, yerr=rmsd_stds, capsize=5, alpha=0.7, color='orange')
    ax2.set_title('RMSD Comparison')
    ax2.set_ylabel('Root Mean Square Deviation')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Time
    ax3.bar(strategies, time_means, yerr=time_stds, capsize=5, alpha=0.7, color='green')
    ax3.set_title('Computation Time Comparison')
    ax3.set_ylabel('Time (seconds)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def run_detailed_comparison():
    """Run a series of tests for different parameters."""
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8]
    # Test with different sequence lengths
    seq_lengths = [10, 50, 100]
    # Test with different perturbation scales
    perturbation_scales = [1, 4, 10]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on {device}")
    
    # Store all results
    all_results = {}
    
    # Different batch sizes
    print("\n=== Testing with different batch sizes ===")
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        results = compare_alignment_strategies(
            batch_size=batch_size, 
            seq_len=50,
            num_atoms=3, 
            perturbation_scale=4, 
            runs=5,
            device=device
        )
        print_results(results)
        all_results[f"batch_size_{batch_size}"] = results
    
    # Different sequence lengths
    print("\n=== Testing with different sequence lengths ===")
    for seq_len in seq_lengths:
        print(f"\nTesting with sequence length: {seq_len}")
        results = compare_alignment_strategies(
            batch_size=2, 
            seq_len=seq_len,
            num_atoms=3, 
            perturbation_scale=4, 
            runs=5,
            device=device
        )
        print_results(results)
        all_results[f"seq_len_{seq_len}"] = results
    
    # Different perturbation scales
    print("\n=== Testing with different perturbation scales ===")
    for scale in perturbation_scales:
        print(f"\nTesting with perturbation scale: {scale}")
        results = compare_alignment_strategies(
            batch_size=2, 
            seq_len=50,
            num_atoms=3, 
            perturbation_scale=scale, 
            runs=5,
            device=device
        )
        print_results(results)
        all_results[f"scale_{scale}"] = results
    
    return all_results

def run_basic_comparison():
    """Run a basic comparison with default parameters."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on {device}")
    
    results = compare_alignment_strategies(
        batch_size=2, 
        seq_len=50,
        num_atoms=3, 
        perturbation_scale=4, 
        runs=10,
        device=device
    )
    
    print_results(results)
    
    # Create output directory if it doesn't exist
    output_dir = Path('/mnt/hdd8/mehdi/projects/vq_encoder_decoder/tests/outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save results
    plot_results(results, save_path=str(output_dir / 'alignment_comparison.png'))
    
    return results

if __name__ == "__main__":
    print("Testing alignment strategies...")
    
    # Choose which comparison to run
    detailed = False  # Set to True for detailed comparison, False for basic
    
    if detailed:
        all_results = run_detailed_comparison()
    else:
        results = run_basic_comparison()
        
    print("Testing complete!")
