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
                                 runs=5, strategies=None, device='cpu', perturbation_type='both'):
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
        perturbation_type: Type of perturbation to apply ('translation_only', 'rotation_only', 'both')
        
    Returns:
        results: Dictionary containing the results
    """
    if strategies is None:
        strategies = ['kabsch', 'kabsch_old', 'quaternion']

    results = {strategy: {'mse': [], 'rmsd': [], 'time': []} for strategy in strategies}

    for i in range(runs):
        print(f"Run {i + 1}/{runs}")

        # Generate random true coordinates
        x_true = torch.randn(batch_size, seq_len, num_atoms, 3, device=device)

        # Generate masks with padding at the end (1 for valid positions, 0 for padding)
        masks = torch.zeros(batch_size, seq_len, device=device)
        for b in range(batch_size):
            # Random valid length (at least 1, at most seq_len)
            valid_length = torch.randint(1, seq_len + 1, (1,)).item()
            masks[b, :valid_length] = 1

        # Apply perturbation based on the specified type
        x_true_perturbed = x_true.clone()
        
        if perturbation_type in ['rotation_only', 'both']:
            # Apply rotation
            quat = torch.rand(batch_size, 4, device=device)
            rot = quaternion_to_matrix(quat)
            x_true_perturbed = rot.bmm(x_true_perturbed.flatten(1, 2).mT).mT.reshape_as(x_true)
            
        if perturbation_type in ['translation_only', 'both']:
            # Apply translation (uniform random translation for each batch)
            translation = torch.rand(batch_size, 1, 1, 3, device=device) * perturbation_scale
            x_true_perturbed = x_true_perturbed + translation

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
    print("\n" + "=" * 80)
    print(
        f"{'Strategy':<15} | {'MSE Mean':<12} | {'MSE StdDev':<12} | {'RMSD Mean':<12} | {'RMSD StdDev':<12} | {'Time (s)':<12}")
    print("-" * 80)

    for strategy, metrics in results.items():
        print(f"{strategy:<15} | "
              f"{metrics['mse']['mean']:<12.6f} | "
              f"{metrics['mse']['std']:<12.6f} | "
              f"{metrics['rmsd']['mean']:<12.6f} | "
              f"{metrics['rmsd']['std']:<12.6f} | "
              f"{metrics['time']['mean']:<12.6f}")

    print("=" * 80 + "\n")


def plot_results(results, save_path=None, title=None):
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
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=16)

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
    # Test with different perturbation types
    perturbation_types = ['translation_only', 'rotation_only', 'both']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on {device}")

    # Store all results
    all_results = {}

    # Different perturbation types
    print("\n=== Testing with different perturbation types ===")
    for p_type in perturbation_types:
        print(f"\nTesting with perturbation type: {p_type}")
        results = compare_alignment_strategies(
            batch_size=2,
            seq_len=50,
            num_atoms=3,
            perturbation_scale=4,
            runs=5,
            device=device,
            perturbation_type=p_type
        )
        print_results(results)
        all_results[f"perturbation_{p_type}"] = results

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
    
    # Create output directory if it doesn't exist
    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test different perturbation types
    perturbation_types = ['translation_only', 'rotation_only', 'both']
    all_results = {}
    
    for p_type in perturbation_types:
        print(f"\n=== Testing with perturbation type: {p_type} ===")
        results = compare_alignment_strategies(
            batch_size=2,
            seq_len=128,
            num_atoms=3,
            perturbation_scale=4,
            runs=10,
            device=device,
            perturbation_type=p_type
        )
        
        print_results(results)
        all_results[p_type] = results
        
        # Plot and save individual results
        plot_results(
            results, 
            save_path=str(output_dir / f'alignment_comparison_{p_type}.png'),
            title=f'Alignment Comparison with {p_type.replace("_", " ").title()} Perturbation'
        )
    
    # Compare all perturbation types in a combined visualization
    plot_perturbation_comparison(all_results, save_path=str(output_dir / 'perturbation_type_comparison.png'))
    
    return all_results


def plot_perturbation_comparison(all_results, save_path=None):
    """Plot a comparison of different perturbation types."""
    perturbation_types = list(all_results.keys())
    strategies = list(all_results[perturbation_types[0]].keys())
    
    # Create figure with 2 subplots (MSE and RMSD)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    fig.suptitle('Comparison of Alignment Methods with Different Perturbation Types', fontsize=16)
    
    # Set width of bars
    bar_width = 0.25
    r = np.arange(len(strategies))
    
    # Colors for different perturbation types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot MSE
    for i, p_type in enumerate(perturbation_types):
        mse_means = [all_results[p_type][s]['mse']['mean'] for s in strategies]
        mse_stds = [all_results[p_type][s]['mse']['std'] for s in strategies]
        
        pos = [x + bar_width * i for x in r]
        ax1.bar(pos, mse_means, bar_width, label=p_type.replace('_', ' ').title(),
                yerr=mse_stds, capsize=4, alpha=0.7, color=colors[i])
    
    ax1.set_title('MSE by Perturbation Type')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_xticks([x + bar_width for x in r])
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot RMSD
    for i, p_type in enumerate(perturbation_types):
        rmsd_means = [all_results[p_type][s]['rmsd']['mean'] for s in strategies]
        rmsd_stds = [all_results[p_type][s]['rmsd']['std'] for s in strategies]
        
        pos = [x + bar_width * i for x in r]
        ax2.bar(pos, rmsd_means, bar_width, label=p_type.replace('_', ' ').title(),
                yerr=rmsd_stds, capsize=4, alpha=0.7, color=colors[i])
    
    ax2.set_title('RMSD by Perturbation Type')
    ax2.set_ylabel('Root Mean Square Deviation')
    ax2.set_xticks([x + bar_width for x in r])
    ax2.set_xticklabels(strategies)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Testing alignment strategies...")

    # Choose which comparison to run
    detailed = False  # Set to True for detailed comparison, False for basic

    if detailed:
        all_results = run_detailed_comparison()
    else:
        results = run_basic_comparison()

    print("Testing complete!")

