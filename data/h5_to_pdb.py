import argparse
import os
import glob
import torch
from tqdm import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the parent directory to the Python path to allow imports from 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import load_h5_file, save_backbone_pdb_inference


def find_h5_files(directory_path):
    """
    Finds all .h5 files in the given directory and its subdirectories.
    """
    h5_files_pattern = os.path.join(directory_path, '**', '*.h5')
    h5_files = glob.glob(h5_files_pattern, recursive=True)
    return h5_files


def convert_h5_to_pdb(h5_path, pdb_dir):
    """Converts a single h5 file to a PDB file."""
    try:
        seq, n_ca_c_o_coord, _ = load_h5_file(h5_path)

        # We only need N, CA, C coordinates for the backbone PDB.
        backbone_coords = torch.from_numpy(n_ca_c_o_coord[:, :3, :])
        # build mask: skip residues with any NaN coordinates
        mask = (~torch.isnan(backbone_coords).any(dim=(1, 2))).to(dtype=torch.int)

        base_name = os.path.splitext(os.path.basename(h5_path))[0]
        pdb_path = os.path.join(pdb_dir, f"{base_name}.pdb")

        save_backbone_pdb_inference(backbone_coords, mask, pdb_path)
        return True, h5_path, None

    except Exception as e:
        return False, h5_path, str(e)


def main():
    parser = argparse.ArgumentParser(description='Convert h5 files to PDB files.')
    parser.add_argument('--h5_dir', type=str, required=True, help='Directory containing h5 files.')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Directory to save PDB files.')
    parser.add_argument('--max_workers', type=int, default=os.cpu_count(),
                        help='Number of worker processes to use for conversion (default: CPU count).')
    args = parser.parse_args()

    os.makedirs(args.pdb_dir, exist_ok=True)

    h5_files = find_h5_files(args.h5_dir)

    if not h5_files:
        print(f"No .h5 files found in {args.h5_dir}")
        return

    errors = []

    if args.max_workers and args.max_workers > 1:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(convert_h5_to_pdb, h5_file, args.pdb_dir): h5_file
                for h5_file in h5_files
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting h5 to PDB"):
                success, path, message = future.result()
                if not success:
                    errors.append((path, message))
    else:
        for h5_file in tqdm(h5_files, desc="Converting h5 to PDB"):
            success, path, message = convert_h5_to_pdb(h5_file, args.pdb_dir)
            if not success:
                errors.append((path, message))

    if errors:
        print(f"Finished with {len(errors)} errors. See details below:")
        for path, message in errors:
            print(f"  {path}: {message}")
    print(f"Conversion complete. PDB files are saved in {args.pdb_dir}")


if __name__ == '__main__':
    main()
