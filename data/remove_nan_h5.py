import argparse
import os
import h5py
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def contains_nan(h5_filepath):
    """
    Checks if the 'N_CA_C_O_coord' dataset in an HDF5 file contains NaN values.

    Args:
        h5_filepath (str): Path to the HDF5 file.

    Returns:
        bool: True if NaN values are found, False otherwise.
    """
    try:
        with h5py.File(h5_filepath, 'r') as f:
            if 'N_CA_C_O_coord' in f:
                coords = f['N_CA_C_O_coord'][:]
                return np.isnan(coords).any()
            else:
                print(f"Warning: 'N_CA_C_O_coord' dataset not found in {h5_filepath}")
                return False
    except Exception as e:
        print(f"Error reading {h5_filepath}: {e}")
        return False # Treat as no NaN if file is unreadable or corrupt to avoid accidental deletion

def main():
    parser = argparse.ArgumentParser(description="Remove HDF5 files containing NaN coordinates.")
    parser.add_argument('--h5_dir', type=str, default='./save_test/',
                        help="Directory containing HDF5 files. Defaults to './save_test/'.")
    parser.add_argument('--max_workers', type=int, default=16,
                        help="Number of worker processes to use. Defaults to 16.")
    args = parser.parse_args()

    if not os.path.isdir(args.h5_dir):
        print(f"Error: Directory not found: {args.h5_dir}")
        return

    h5_files_pattern = os.path.join(args.h5_dir, '*.h5')
    h5_files = glob.glob(h5_files_pattern)

    if not h5_files:
        print(f"No .h5 files found in {args.h5_dir}")
        return

    files_to_remove = []
    print(f"Checking {len(h5_files)} files for NaN coordinates using up to {args.max_workers} workers...")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {executor.submit(contains_nan, h5_file): h5_file for h5_file in h5_files}
        for future in tqdm(as_completed(future_to_file), total=len(h5_files), desc="Checking files"):
            h5_file = future_to_file[future]
            try:
                if future.result():  # contains_nan returned True
                    files_to_remove.append(h5_file)
            except Exception as exc:
                print(f"An error occurred while checking {h5_file}: {exc}")

    removed_count = 0
    if files_to_remove:
        print(f"\nFound {len(files_to_remove)} files with NaN coordinates. Attempting to remove them...")
        for h5_file in tqdm(files_to_remove, desc="Removing files"):
            try:
                os.remove(h5_file)
                # print(f"Removed: {h5_file}") # tqdm will show progress
                removed_count += 1
            except OSError as e:
                print(f"Error deleting {h5_file}: {e}")
    else:
        print("\nNo files with NaN coordinates found to remove.")

    print(f"\nFinished processing. Removed {removed_count} out of {len(files_to_remove)} identified files.")

if __name__ == '__main__':
    main()

