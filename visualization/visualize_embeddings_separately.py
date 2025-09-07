import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from visualization.utils import find_h5_file_in_dir, load_embeddings_from_h5


def spectrum_color(t):
    """Return an RGB tuple on a visible-spectrum-like gradient from purple->red for t in [0,1].

    This uses several hand-picked color stops and linearly interpolates between them.
    """
    # Color stops approximating the visible spectrum (purple -> indigo -> blue -> cyan -> green -> yellow -> orange -> red)
    stops = [
        (128, 0, 128),   # purple
        (75, 0, 130),    # indigo / deep purple
        (0, 0, 255),     # blue
        (0, 255, 255),   # cyan
        (0, 255, 0),     # green
        (255, 255, 0),   # yellow
        (255, 127, 0),   # orange
        (255, 0, 0),     # red
    ]
    stops = [tuple(np.array(c) / 255.0) for c in stops]
    t = float(np.clip(t, 0.0, 1.0))
    n = len(stops)
    if t >= 1.0:
        return stops[-1]
    pos = t * (n - 1)
    i = int(np.floor(pos))
    frac = pos - i
    c0 = np.array(stops[i])
    c1 = np.array(stops[min(i + 1, n - 1)])
    return tuple((1 - frac) * c0 + frac * c1)


def plot_protein_residues(Y, pid, out_dir, start_color=(0.5, 0.0, 0.5), end_color=(1.0, 0.0, 0.0)):
    """Plot residues for a single protein.

    Y: (L, 2) array of projected points for the residues of this protein
    pid: protein id (used for filename and title)
    out_dir: where to save the figure
    start_color: RGB color for first residue (purple by default)
    end_color: RGB color for last residue (red by default)
    """
    L = Y.shape[0]
    if L == 0:
        return

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Vectorized scatter: prepare per-residue colors then draw each layer once (much faster)
    t_vals = np.arange(L) / max(1, L - 1)
    colors = [spectrum_color(float(t)) for t in t_vals]
    xs = Y[:, 0]
    ys = Y[:, 1]

    # larger, faint halo (rasterize for faster file sizes if complex)
    ax.scatter(xs, ys, s=160, c=colors, alpha=0.12, linewidths=0, rasterized=True)
    # slightly smaller, mid halo
    ax.scatter(xs, ys, s=80, c=colors, alpha=0.22, linewidths=0, rasterized=True)
    # main dot
    ax.scatter(xs, ys, s=18, c=colors, edgecolors='k', linewidths=0.3)

    # label residue index near the dot (draw per-point Text objects; this remains per-point)
    dx = 0.0
    dy = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for i in range(L):
        ax.text(xs[i] + dx, ys[i] + dy, str(i), fontsize=6, va='bottom', ha='center')

    ax.set_title(f"Residue embeddings (t-SNE) for {pid}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    fname = f"tsne_{pid}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved per-protein t-SNE plot: {out_path}")


def main():
    # Hardcoded paths and parameters (modify as needed)
    latest_dir = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/inference_embed_results/ablation_2025-09-01__23-06-32/casp16/2025-09-07__17-58-00"
    out_dir = "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/results/dgx/ablation/2025-09-01__23-06-32/plots/casp16/plots_per_protein"
    os.makedirs(out_dir, exist_ok=True)

    perplexity = 30
    n_iter = 1000
    start_color = (0.5, 0.0, 0.5)  # purple
    end_color = (1.0, 0.0, 0.0)    # red

    h5_path = find_h5_file_in_dir(latest_dir)
    if h5_path is None:
        raise FileNotFoundError(f"No HDF5 file found under {latest_dir}")

    embeddings = load_embeddings_from_h5(h5_path)

    # Build consolidated array for t-SNE: concatenate all residues from all proteins
    pids = []
    points = []
    pid_to_range = {}
    idx = 0
    for pid, emb in embeddings.items():
        L = emb.shape[0]
        if L == 0:
            pid_to_range[pid] = (idx, idx)
            continue
        points.append(emb)
        pid_to_range[pid] = (idx, idx + L)
        idx += L
        pids.append(pid)

    if len(points) == 0:
        raise RuntimeError("No embeddings found in HDF5 file")

    X = np.vstack(points)

    # Basic perplexity check
    n_samples = X.shape[0]
    if perplexity >= n_samples / 3:
        p = max(5, int(n_samples / 5))
        print(f"Adjusting perplexity from {perplexity} to {p} (n_samples={n_samples})")
        perplexity = p

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init='pca', random_state=42)
    Y_all = tsne.fit_transform(X)

    # Now iterate over proteins and save per-protein plots using the corresponding slice of Y_all
    # We'll iterate over pid_to_range in insertion order but ensure we use the correct pids list
    # pids contains only those appended to points; some embeddings with L==0 are skipped there
    offset = 0
    for pid in embeddings.keys():
        start, end = pid_to_range.get(pid, (None, None))
        if start is None or end is None or start == end:
            print(f"Skipping {pid} (no residues)")
            continue
        Y = Y_all[start:end]
        plot_protein_residues(Y, pid, out_dir, start_color=start_color, end_color=end_color)


if __name__ == '__main__':
    main()


