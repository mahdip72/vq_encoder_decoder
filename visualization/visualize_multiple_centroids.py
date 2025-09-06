import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import random
from visualization.utils import find_h5_file_in_dir, load_embeddings_from_h5, find_all_h5_under


def random_color():
    return (random.random(), random.random(), random.random())


def main():
    # Hardcoded root containing dated result subdirectories
    results_root = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/inference_embed_results/test_set_b_2048_h5"
    output_base_dir = "./visualization/plots"
    os.makedirs(output_base_dir, exist_ok=True)

    perplexity = 30
    n_iter = 1000

    # Recursively gather all .h5 files under results_root using utility
    h5_paths = find_all_h5_under(results_root)

    if not h5_paths:
        raise FileNotFoundError(f"No HDF5 files found under {results_root}")

    all_centroids = []
    origin_labels = []  # which h5 (entry) each centroid comes from
    file_colors = {}

    for entry_name, h5p in h5_paths:
        embeddings = load_embeddings_from_h5(h5p)
        color = random_color()
        file_colors[entry_name] = color
        for pid, emb in embeddings.items():
            # emb shape (L, D) -> centroid shape (D,)
            if emb.size == 0:
                continue
            ctr = np.mean(emb, axis=0)
            all_centroids.append(ctr)
            origin_labels.append(entry_name)

    X = np.asarray(all_centroids)
    if X.shape[0] == 0:
        raise RuntimeError("No centroids to visualize")

    # Run t-SNE on centroids
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init='random', random_state=42)
    Y = tsne.fit_transform(X)

    # Plot centroids colored per h5 file
    plt.figure(figsize=(10, 10))
    unique_files = sorted(set(origin_labels))
    for fname in unique_files:
        idx = [i for i, l in enumerate(origin_labels) if l == fname]
        pts = Y[idx]
        c = file_colors[fname]
        plt.scatter(pts[:, 0], pts[:, 1], c=[c], label=fname, s=5, alpha=0.8)

    # plt.legend(loc='best', markerscale=2, fontsize='small')
    out_path = os.path.join(output_base_dir, f"tsne_multi_centroids.svg")
    plt.title(f"t-SNE centroids per HDF5 file under {os.path.basename(results_root)}")
    plt.savefig(out_path)
    print(f"Saved multi-file t-SNE centroids plot to: {out_path}")


if __name__ == '__main__':
    main()


