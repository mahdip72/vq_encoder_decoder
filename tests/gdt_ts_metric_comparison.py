import sys
import numpy as np
import torch
from pathlib import Path

# make sure we can import our metrics
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.metrics import GDTTS, GDTTSNew


def compare_metrics(preds, target, masks=None):
    """Compute both metrics on a single batch, applying mask to GDTTSNew."""
    # Ensure metrics are on the same device as the data
    device = preds.device
    old = GDTTS().to(device)
    # GDTTSNew can handle masks
    new = GDTTSNew().to(device)

    # Update metrics
    old.update(preds, target) # Original GDTTS does not accept masks
    new.update(preds, target, masks=masks) # GDTTSNew uses the mask

    return old.compute().item(), new.compute().item()


def generate_data_with_mask(batch_size, seq_len, noise, device):
    """Generates target, prediction, and a mask for varying sequence lengths."""
    target = torch.randn(batch_size, seq_len, 3, device=device)
    pred = target + torch.randn_like(target) * noise

    # Create masks with random valid lengths for each sample in the batch
    masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        # Ensure at least one residue is valid, max is seq_len
        valid_len = torch.randint(low=max(1, seq_len // 2), high=seq_len + 1, size=(1,)).item()
        masks[i, :valid_len] = True

    return pred, target, masks


def scenario_noise_levels(batch_size=4, seq_len=50, noise_levels=(0.5, 1.0, 2.0, 5.0), runs=10, device='cpu'):
    results = {'noise': [], 'old_mean': [], 'new_mean': []}
    print(f"Running noise scenario with seq_len={seq_len}, batch_size={batch_size}")
    for noise in noise_levels:
        vals_old, vals_new = [], []
        for _ in range(runs):
            # Generate data with masks
            pred, target, masks = generate_data_with_mask(batch_size, seq_len, noise, device)
            # Compare metrics, passing mask to compare_metrics
            o, n = compare_metrics(pred, target, masks=masks)
            vals_old.append(o)
            vals_new.append(n)
        results['noise'].append(noise)
        results['old_mean'].append(np.mean(vals_old))
        results['new_mean'].append(np.mean(vals_new))
    return results


def scenario_batch_sizes(batch_sizes=(1, 2, 8, 16), seq_len=50, noise=1.0, runs=10, device='cpu'):
    results = {'batch': [], 'old_mean': [], 'new_mean': []}
    print(f"Running batch size scenario with seq_len={seq_len}, noise={noise}")
    for b in batch_sizes:
        vals_old, vals_new = [], []
        for _ in range(runs):
             # Generate data with masks
            pred, target, masks = generate_data_with_mask(b, seq_len, noise, device)
            # Compare metrics, passing mask to compare_metrics
            o, n = compare_metrics(pred, target, masks=masks)
            vals_old.append(o);
            vals_new.append(n)
        results['batch'].append(b)
        results['old_mean'].append(np.mean(vals_old))
        results['new_mean'].append(np.mean(vals_new))
    return results


def scenario_seq_lengths(batch_size=4, seq_lens=(10, 50, 100, 200), noise=1.0, runs=10, device='cpu'):
    results = {'seq_len': [], 'old_mean': [], 'new_mean': []}
    print(f"Running sequence length scenario with batch_size={batch_size}, noise={noise}")
    for L in seq_lens:
        vals_old, vals_new = [], []
        for _ in range(runs):
             # Generate data with masks
            pred, target, masks = generate_data_with_mask(batch_size, L, noise, device)
            # Compare metrics, passing mask to compare_metrics
            o, n = compare_metrics(pred, target, masks=masks)
            vals_old.append(o);
            vals_new.append(n)
        results['seq_len'].append(L)
        results['old_mean'].append(np.mean(vals_old))
        results['new_mean'].append(np.mean(vals_new))
    return results


def print_table(label, results, xkey):
    print(f"\n--- {label} ---")
    hdr = f"{xkey:>10} | {'GDTTS_old':>10} | {'GDTTS_new':>10}"
    print(hdr);
    print("-" * len(hdr))
    for x, o, n in zip(results[xkey], results['old_mean'], results['new_mean']):
        print(f"{x:>10} | {o:10.4f} | {n:10.4f}")


if __name__ == "__main__":
    # Force CPU for all operations
    device = 'cpu'
    print(f"Running GDT-TS comparison on device: {device}")
    print("Introducing masks to simulate padding...")

    # Define common parameters
    base_batch_size = 4
    base_seq_len = 100
    base_noise = 1.0
    num_runs = 20 # Increase runs for potentially more stable averages

    # compare over noise levels
    noise_res = scenario_noise_levels(
        batch_size=base_batch_size,
        seq_len=base_seq_len,
        runs=num_runs,
        device=device
    )
    print_table("Varying Noise (with Masks)", noise_res, 'noise')

    # compare over batch sizes
    batch_res = scenario_batch_sizes(
        seq_len=base_seq_len,
        noise=base_noise,
        runs=num_runs,
        device=device
    )
    print_table("Varying Batch Size (with Masks)", batch_res, 'batch')

    # compare over sequence lengths
    seq_res = scenario_seq_lengths(
        batch_size=base_batch_size,
        noise=base_noise,
        runs=num_runs,
        device=device
    )
    print_table("Varying Sequence Length (with Masks)", seq_res, 'seq_len')

    print("\nComparison finished. Note the differences, especially with varying sequence lengths and masks.")
    print("GDTTSNew uses masks correctly, while GDTTS (old) includes padded regions, potentially skewing results.")
