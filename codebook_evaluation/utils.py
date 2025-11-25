import os
import h5py
import numpy as np
from typing import Dict, Tuple


def find_latest_results_dir(results_root: str) -> str:
    """Return the most recently modified subdirectory inside results_root.

    If results_root itself contains HDF5 files, returns results_root.
    """
    if not os.path.exists(results_root):
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    # If results_root contains any .h5 file, assume it's the target
    for el in os.listdir(results_root):
        if el.endswith('.h5'):
            return results_root

    subdirs = [os.path.join(results_root, d) for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in results root: {results_root}")

    latest = max(subdirs, key=lambda p: os.path.getmtime(p))
    return latest


def find_h5_file_in_dir(directory: str, pattern: str = None) -> str:
    """Return path to the first .h5 file in directory. If pattern provided, prefer files that contain pattern."""
    files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {directory}")
    if pattern:
        for f in files:
            if pattern in f:
                return os.path.join(directory, f)
    # fallback to newest .h5 by modification time
    files_full = [os.path.join(directory, f) for f in files]
    latest = max(files_full, key=lambda p: os.path.getmtime(p))
    return latest


def load_embeddings_from_h5(h5_path: str) -> Dict[str, np.ndarray]:
    """Load per-pid embeddings stored as datasets in an HDF5 file.

    Returns a dict mapping pid -> ndarray shape (L, D)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    out = {}
    with h5py.File(h5_path, 'r') as hf:
        for key in hf.keys():
            data = hf[key]['embedding'][:]
            out[key] = np.asarray(data)
    return out


def find_all_h5_under(results_root: str) -> list:
    """Recursively find all .h5 files under results_root and return list of (label, path)

    Label is the immediate parent folder name (useful for dated result directories).
    """
    if not os.path.exists(results_root):
        raise FileNotFoundError(results_root)

    h5_paths = []
    for root, dirs, files in os.walk(results_root):
        for f in files:
            if f.endswith('.h5'):
                full = os.path.join(root, f)
                parent = os.path.basename(os.path.dirname(full))
                label = parent if parent else os.path.basename(full)
                h5_paths.append((label, full))
    return h5_paths


