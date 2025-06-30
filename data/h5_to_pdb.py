import argparse
import os
import glob
import torch
from tqdm import tqdm
import sys

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
    """
    Converts a single h5 file to a PDB file.
    """
    try:
        seq, n_ca_c_o_coord, _ = load_h5_file(h5_path)

        # We only need N, CA, C coordinates for the backbone PDB.
        backbone_coords = torch.from_numpy(n_ca_c_o_coord[:, :3, :])
        seq_len = backbone_coords.shape[0]
        mask = torch.ones(seq_len, dtype=torch.int)

        base_name = os.path.splitext(os.path.basename(h5_path))[0]
        pdb_path = os.path.join(pdb_dir, f"{base_name}.pdb")

        save_backbone_pdb_inference(backbone_coords, mask, pdb_path)

    except Exception as e:
        print(f"Could not process {h5_path}. Reason: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert h5 files to PDB files.')
    parser.add_argument('--h5_dir', type=str, required=True, help='Directory containing h5 files.')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Directory to save PDB files.')
    args = parser.parse_args()

    os.makedirs(args.pdb_dir, exist_ok=True)

    h5_files = find_h5_files(args.h5_dir)

    if not h5_files:
        print(f"No .h5 files found in {args.h5_dir}")
        return

    for h5_file in tqdm(h5_files, desc="Converting h5 to PDB"):
        convert_h5_to_pdb(h5_file, args.pdb_dir)

    print(f"Conversion complete. PDB files are saved in {args.pdb_dir}")


if __name__ == '__main__':
    main()

