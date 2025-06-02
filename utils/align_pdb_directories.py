import os
import argparse
import numpy as np
import torch
from graphein.protein.tensor.geometry import kabsch
import re


def parse_pdb_backbone_coordinates(pdb_file_path):
    """
    Parses a PDB file to extract backbone (N, CA, C) atom coordinates and all lines.

    Args:
        pdb_file_path (str): Path to the PDB file.

    Returns:
        tuple: (list of str, np.ndarray, list of int)
            - lines: All lines from the PDB file.
            - backbone_coords: NumPy array of N, CA, C coordinates (M, 3), where M is total backbone atoms.
            - backbone_line_indices: List of line indices for N, CA, C ATOM records.
            Returns (None, None, None) if file not found or parsing error.
    """
    if not os.path.exists(pdb_file_path):
        print(f"Warning: PDB file not found: {pdb_file_path}")
        return None, None, None

    lines = []
    backbone_coords_list = []
    backbone_line_indices = []
    valid_atom_types = {"N", "CA", "C"}

    try:
        with open(pdb_file_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name in valid_atom_types:
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        backbone_coords_list.append([x, y, z])
                        backbone_line_indices.append(i)
                    except ValueError:
                        print(f"Warning: Could not parse coordinates for a {atom_name} atom in {pdb_file_path} at line {i + 1}")
                        continue  # Skip this atom if coords are malformed

        if not backbone_coords_list:
            # print(f"Warning: No backbone (N, CA, C) atoms found in {pdb_file_path}") # Can be verbose
            return lines, np.array([]).reshape(0, 3), []

        return lines, np.array(backbone_coords_list, dtype=np.float32), backbone_line_indices
    except Exception as e:
        print(f"Error reading or parsing PDB file {pdb_file_path}: {e}")
        return None, None, None


def write_aligned_pdb(original_lines, backbone_line_indices, aligned_backbone_coords, output_pdb_path):
    """
    Writes a new PDB file with updated backbone (N, CA, C) atom coordinates.

    Args:
        original_lines (list of str): Original lines from the PDB file.
        backbone_line_indices (list of int): Line indices of N, CA, C ATOM records.
        aligned_backbone_coords (np.ndarray): NumPy array of aligned N, CA, C coordinates (M, 3).
        output_pdb_path (str): Path to save the new PDB file.
    """
    modified_lines = list(original_lines)  # Make a copy

    if len(backbone_line_indices) != len(aligned_backbone_coords):
        print(
            f"Error: Mismatch between number of backbone atom line indices ({len(backbone_line_indices)}) and aligned coordinates ({len(aligned_backbone_coords)}) for {output_pdb_path}. Skipping write.")
        return

    for i, line_idx in enumerate(backbone_line_indices):
        if i >= len(aligned_backbone_coords):
            print(
                f"Warning: More backbone line indices than aligned coordinates. Stopping at index {i} for {output_pdb_path}")
            break

        coords = aligned_backbone_coords[i]
        # Format coordinates to PDB specification (F8.3)
        x_str = f"{coords[0]:8.3f}"
        y_str = f"{coords[1]:8.3f}"
        z_str = f"{coords[2]:8.3f}"

        # Ensure strings are not too long for the fixed width format
        x_str = x_str[:8]
        y_str = y_str[:8]
        z_str = z_str[:8]

        line = modified_lines[line_idx]
        # Replace coordinate part of the line
        # ATOM line: "ATOM ... XXXXXXXX YYYYYYYY ZZZZZZZZ ..."
        # Indices:    0    ... 30-37    38-45    46-53
        modified_lines[line_idx] = line[:30] + x_str + y_str + z_str + line[54:]

    try:
        with open(output_pdb_path, 'w') as f:
            f.writelines(modified_lines)
        # print(f"Successfully wrote aligned PDB to: {output_pdb_path}")
    except Exception as e:
        print(f"Error writing aligned PDB file {output_pdb_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Align PDB files from two directories using Kabsch alignment on backbone (N, CA, C) atoms.")
    parser.add_argument("--dir1", help="Path to the first directory containing PDB files (e.g., predictions).")
    parser.add_argument("--dir2", help="Path to the second directory containing PDB files (e.g., ground truth).")
    parser.add_argument("--output_dir", help="Path to the directory to save aligned PDB files.")
    # Removed --atom_type argument as we are now hardcoding to N, CA, C

    args = parser.parse_args()

    if not os.path.isdir(args.dir1):
        print(f"Error: Directory not found: {args.dir1}")
        return
    if not os.path.isdir(args.dir2):
        print(f"Error: Directory not found: {args.dir2}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Aligning PDBs from '{args.dir1}' to '{args.dir2}' using N, CA, C backbone atoms. Outputting to '{args.output_dir}'.")
    processed_files = 0
    aligned_files_count = 0

    pdb_files_dir1 = sorted([f for f in os.listdir(args.dir1) if f.endswith(".pdb")])

    # Updated regex to match new filenames with sample index
    dir1_file_pattern = re.compile(r"valid_outputs_epoch_\d+_(step_\d+)_sample_(\d+)\.pdb")

    for pdb_file_name_dir1 in pdb_files_dir1:
        processed_files += 1
        path1 = os.path.join(args.dir1, pdb_file_name_dir1)

        match = dir1_file_pattern.match(pdb_file_name_dir1)
        if not match:
            print(f"Warning: Filename {pdb_file_name_dir1} in dir1 does not match expected pattern 'valid_outputs_epoch_E_step_S_sample_I.pdb'. Skipping.")
            continue

        step_part = match.group(1)  # e.g., 'step_1'
        sample_idx = match.group(2) # e.g., '0'
        # Construct the corresponding filename for dir2
        pdb_file_name_dir2 = f"valid_labels_{step_part}_sample_{sample_idx}.pdb"
        path2 = os.path.join(args.dir2, pdb_file_name_dir2)

        if not os.path.exists(path2):
            print(f"Warning: Matching PDB file {pdb_file_name_dir2} not found in dir2 for {pdb_file_name_dir1}. Skipping.")
            continue

        lines1, coords1_np, indices1 = parse_pdb_backbone_coordinates(path1)
        lines2, coords2_np, indices2 = parse_pdb_backbone_coordinates(path2)

        if lines1 is None or coords1_np is None:
            print(f"Skipping {pdb_file_name_dir1} due to parsing error in dir1.")
            continue
        if coords2_np is None:
            print(f"Skipping {pdb_file_name_dir1} (paired with {pdb_file_name_dir2}) due to parsing error or no backbone atoms in corresponding file in dir2.")
            continue

        if coords1_np.shape[0] == 0:
            print(f"Warning: No backbone (N, CA, C) atoms found in {path1}. Skipping.")
            continue
        if coords2_np.shape[0] == 0:
            print(f"Warning: No backbone (N, CA, C) atoms found in {path2}. Skipping.")
            continue

        if coords1_np.shape[0] != coords2_np.shape[0]:
            print(
                f"Warning: Mismatch in backbone atom count for {pdb_file_name_dir1} and {pdb_file_name_dir2} ({coords1_np.shape[0]} vs {coords2_np.shape[0]}). Skipping.")
            continue

        if coords1_np.shape[0] == 0:
            print(f"Warning: Zero backbone atoms to align for {pdb_file_name_dir1}. Skipping.")
            continue

        coords1_torch = torch.from_numpy(coords1_np).float()
        coords2_torch = torch.from_numpy(coords2_np).float()

        try:
            aligned_coords1_torch = kabsch(coords1_torch, coords2_torch, allow_reflections=True)
            aligned_coords1_np = aligned_coords1_torch.detach().cpu().numpy()
        except Exception as e:
            print(f"Error during Kabsch alignment for {pdb_file_name_dir1} and {pdb_file_name_dir2}: {e}. Skipping.")
            continue

        output_file_path = os.path.join(args.output_dir, f"aligned_{pdb_file_name_dir1}")
        write_aligned_pdb(lines1, indices1, aligned_coords1_np, output_file_path)
        aligned_files_count += 1

    print(f"Finished processing. Total files checked in dir1: {len(pdb_files_dir1)}.")
    print(f"Successfully aligned and saved: {aligned_files_count} PDB files.")


if __name__ == "__main__":
    main()
