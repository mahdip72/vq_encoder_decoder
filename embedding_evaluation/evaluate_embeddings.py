import os
import numpy as np
from sklearn.manifold import TSNE  # This import can be removed if not used elsewhere
from embedding_evaluation.utils import find_h5_file_in_dir, load_embeddings_from_h5


def compute_consecutive_metrics(emb):
    """Compute metrics for a single protein's embeddings.

    emb: (L, D) array of original embeddings for the residues
    Returns a dict with metrics or None if L < 2
    """
    L = emb.shape[0]
    if L < 2:
        return None

    # Differences between consecutive residues
    diffs = np.diff(emb, axis=0)  # (L-1, D)
    dists = np.linalg.norm(diffs, axis=1)  # (L-1,) Euclidean distances in original space

    # Metric 1: Average Euclidean Distance Between Consecutive Residues
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)  # Optional, but useful
    max_dist = np.max(dists)  # Optional

    # Metric 2: Total Path Length (Cumulative Distance)
    path_length = np.sum(dists)

    # Optional: End-to-end distance and tortuosity
    end_to_end = np.linalg.norm(emb[-1] - emb[0])
    tortuosity = path_length / end_to_end if end_to_end > 0 else 0

    return {
        'mean_consecutive_dist': mean_dist,
        'std_consecutive_dist': std_dist,
        'max_consecutive_dist': max_dist,
        'path_length': path_length,
        'tortuosity': tortuosity
    }


def main():
    # Hardcoded paths and parameters (modify as needed)
    latest_dir = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/inference_embed_results/ablation_2025-09-05__00-09-51/casp16/2025-09-07__17-48-59"
    out_dir = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/results/dgx/ablation/2025-09-05__00-09-51/plots/casp16/plots_per_protein"
    os.makedirs(out_dir, exist_ok=True)

    # Removed perplexity and n_iter since no t-SNE

    h5_path = find_h5_file_in_dir(latest_dir)
    if h5_path is None:
        raise FileNotFoundError(f"No HDF5 file found under {latest_dir}")

    embeddings = load_embeddings_from_h5(h5_path)

    # No need to build consolidated array for t-SNE

    # Collect metrics across all proteins
    all_metrics = []
    total_path_length = 0.0
    total_consecutive_count = 0
    total_sum_dist = 0.0

    for pid, emb in embeddings.items():
        L = emb.shape[0]
        if L < 2:
            print(f"Skipping {pid} (fewer than 2 residues)")
            continue

        metrics = compute_consecutive_metrics(emb)
        if metrics:
            print(f"Metrics for {pid}: {metrics}")
            all_metrics.append(metrics)

            # Accumulate for averages
            total_path_length += metrics['path_length']
            consec_count = L - 1
            total_consecutive_count += consec_count
            total_sum_dist += metrics['mean_consecutive_dist'] * consec_count

    if all_metrics:
        # Global averages
        avg_mean_dist = total_sum_dist / total_consecutive_count if total_consecutive_count > 0 else 0
        avg_path_length = total_path_length / len(all_metrics)  # Per-protein average, or adjust as needed
        # Could compute average tortuosity, etc.: avg_tort = np.mean([m['tortuosity'] for m in all_metrics])

        print("\nGlobal Averages:")
        print(f"Average Mean Consecutive Distance: {avg_mean_dist}")
        print(f"Average Path Length (per protein): {avg_path_length}")

        # Optionally save to file
        # import json
        # with open(os.path.join(out_dir, 'metrics_summary.json'), 'w') as f:
        #     json.dump({'global_averages': {'avg_mean_dist': avg_mean_dist, 'avg_path_length': avg_path_length},
        #                'per_protein': all_metrics}, f)


if __name__ == '__main__':
    main()