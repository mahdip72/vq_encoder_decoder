import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import random
from visualization.utils import find_h5_file_in_dir, load_embeddings_from_h5


def random_color():
    return (random.random(), random.random(), random.random())


def main():
    # Hardcoded paths and parameters (modify as needed)
    latest_dir = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/inference_embed_results/2025-09-05__17-32-53"  # folder containing the HDF5 file
    output_base_dir = "visualization/plots"
    os.makedirs(output_base_dir, exist_ok=True)
    perplexity = 30
    n_iter = 1000
    # When True: run t-SNE on all residue embeddings but overlay the per-protein centroid (mean) on the 2D map
    overlay_protein_centroids = True

    h5_path = find_h5_file_in_dir(latest_dir)

    embeddings_dict = load_embeddings_from_h5(h5_path)

    # Build a consolidated list of embeddings and labels (pid)
    all_points = []
    labels = []
    color_map = {}

    for pid, emb in embeddings_dict.items():
        # emb is (L, D)
        for v in emb:
            all_points.append(v)
            labels.append(pid)
        color_map[pid] = random_color()

    X = np.asarray(all_points)

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init='random', random_state=42)
    Y = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10))

    if overlay_protein_centroids:
        # Compute and plot only per-protein centroids (mean of projected points)
        centroids = {}
        for pid in set(labels):
            idx = [i for i, l in enumerate(labels) if l == pid]
            pts = Y[idx]
            centroids[pid] = pts.mean(axis=0)

        for pid, ctr in centroids.items():
            c = color_map[pid]
            plt.scatter(ctr[0], ctr[1], c=[c], s=120, edgecolors='k', linewidths=0.8)
            plt.text(ctr[0], ctr[1], pid, fontsize=6, va='center', ha='center')
    else:
        # Plot all residue points
        for pid in set(labels):
            idx = [i for i, l in enumerate(labels) if l == pid]
            pts = Y[idx]
            c = color_map[pid]
            plt.scatter(pts[:, 0], pts[:, 1], c=[c], label=pid, s=5, alpha=0.6)

            plt.legend(loc='best', markerscale=3, fontsize='small')
    
    out_name = os.path.basename(latest_dir.rstrip('/'))
    suffix = "_centroids" if overlay_protein_centroids else ""
    out_path = os.path.join(output_base_dir, f"tsne_{out_name}{suffix}.png")
    plt.title(f"t-SNE embeddings: {out_name}{suffix}")
    plt.savefig(out_path, dpi=200)
    print(f"Saved t-SNE plot to: {out_path}")


if __name__ == '__main__':
    main()


