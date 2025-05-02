import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def _load_and_get_length(path):
    """Loads a single H5 file and returns the sequence length."""
    try:
        # Assuming load_h5_file returns (sequence, coordinates, plddt)
        seq, *_ = load_h5_file(path)
        # Decode if bytes
        if isinstance(seq, (bytes, bytearray)):
            seq = seq.decode("utf-8")
        return len(seq)
    except Exception as e:
        print(f"Error processing file {path}: {e}", file=sys.stderr)
        return None

def get_sequence_lengths(data_dir, max_workers=None):
    """Reads all .h5 files in a directory using multiple threads and returns a list of sequence lengths."""
    h5_paths = glob.glob(os.path.join(data_dir, '**', '*.h5'), recursive=True)
    if not h5_paths:
        print(f"Warning: No .h5 files found in {data_dir}")
        return []

    lengths = []
    print(f"Found {len(h5_paths)} H5 files. Processing...")

    # Use ThreadPoolExecutor for parallel file reading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(_load_and_get_length, path): path for path in h5_paths}

        # Process results as they complete with tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(h5_paths), desc="Reading H5 files"):
            length = future.result()
            if length is not None:
                lengths.append(length)

    if not lengths:
         print("Warning: No valid sequences found after processing.")

    return lengths

def plot_length_distribution(lengths, bins=50, output_path=None, title="Distribution of Amino Acid Sequence Lengths"):
    """Plots a histogram of the sequence lengths."""
    if not lengths:
        print("No sequence lengths to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bins, color="skyblue", edgecolor="black")
    plt.xlabel("Sequence Length (Number of Amino Acids)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)

    # Add statistics text
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)
    std_dev = np.std(lengths)
    stats_text = f'Mean: {mean_len:.2f}\nMedian: {median_len:.0f}\nStd Dev: {std_dev:.2f}\nN: {len(lengths)}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))


    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    DATA_DIR = "/home/mpngf/datasets/vqvae/swissprot_1024_h5/"  # Hardcoded path to the data directory
    BINS = 100
    OUTPUT_PATH = "sequence_length_distribution_swissprot.png"  # Set to None to display instead of saving
    PLOT_TITLE = "Distribution of Amino Acid Sequence Lengths (Training Set)"
    MAX_WORKERS = os.cpu_count()  # Use number of CPU cores, adjust if needed

    # Make sure the utils module can be found if running script directly
    # Add the parent directory (project root) to the Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)
    from utils.utils import load_h5_file # Re-import after path modification

    # Use the hardcoded DATA_DIR
    data_dir = DATA_DIR
    print(f"Using data directory: {data_dir}")

    # Ensure the data directory exists
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Get sequence lengths and plot
    sequence_lengths = get_sequence_lengths(data_dir, max_workers=MAX_WORKERS)
    plot_length_distribution(sequence_lengths, bins=BINS, output_path=OUTPUT_PATH, title=PLOT_TITLE)

