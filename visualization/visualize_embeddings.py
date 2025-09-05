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
    latest_dir = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/inference_embed_results/2025-09-05__16-51-48"  # root directory containing dated result subdirs
    output_base_dir = "visualization/plots"
    os.makedirs(output_base_dir, exist_ok=True)
    perplexity = 30
    n_iter = 1000

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

    # Plot
    plt.figure(figsize=(10, 10))
    for pid in set(labels):
        idx = [i for i, l in enumerate(labels) if l == pid]
        pts = Y[idx]
        c = color_map[pid]
        plt.scatter(pts[:, 0], pts[:, 1], c=[c], label=pid, s=5)

    plt.legend(loc='best', markerscale=3, fontsize='small')
    out_name = os.path.basename(latest_dir.rstrip('/'))
    out_path = os.path.join(output_base_dir, f"tsne_{out_name}.png")
    plt.title(f"t-SNE embeddings: {out_name}")
    plt.savefig(out_path, dpi=200)
    print(f"Saved t-SNE plot to: {out_path}")


if __name__ == '__main__':
    main()


