import torch
import numpy as np
import os
import glob
from typing import List, Dict, Tuple, Optional
from alignment import kabsch
import math
from pathlib import Path
from tqdm import tqdm


def parse_pdb_backbone(pdb_file: str) -> Optional[torch.Tensor]:
    """
    Parse PDB file and extract backbone coordinates (N, CA, C atoms only).

    Args:
        pdb_file: Path to PDB file

    Returns:
        CoordTensor of shape (n_residues*3, 3) with backbone atoms in order [N, CA, C] for each residue
        Returns None if parsing fails or no backbone atoms found
    """
    backbone_atoms = {'N': 0, 'CA': 1, 'C': 2}  # Standard atom indices

    try:
        residue_coords = {}

        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    if atom_name in backbone_atoms:
                        res_num = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())

                        if res_num not in residue_coords:
                            residue_coords[res_num] = {}

                        residue_coords[res_num][atom_name] = [x, y, z]

        if not residue_coords:
            return None

        # Filter residues that have all three backbone atoms
        complete_residues = []
        for res_num in sorted(residue_coords.keys()):
            if all(atom in residue_coords[res_num] for atom in backbone_atoms.keys()):
                complete_residues.append(res_num)

        if len(complete_residues) < 3:  # Need at least 3 residues for meaningful alignment
            return None

        # Create CoordTensor with backbone atoms in order [N, CA, C] for each residue
        n_residues = len(complete_residues)
        backbone_coords = torch.zeros((n_residues * 3, 3), dtype=torch.float32)

        for i, res_num in enumerate(complete_residues):
            for atom_name, atom_offset in backbone_atoms.items():
                coords = residue_coords[res_num][atom_name]
                coord_idx = i * 3 + atom_offset  # N=0, CA=1, C=2 for each residue
                backbone_coords[coord_idx, :] = torch.tensor(coords, dtype=torch.float32)

        return backbone_coords

    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None


def apply_known_transformation(coords: torch.Tensor,
                               rotation_angle: float = math.pi / 4,
                               rotation_axis: str = 'z',
                               translation: List[float] = [2.0, 3.0, 1.0]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a known rotation and translation to backbone coordinates.

    Args:
        coords: Input backbone coordinates tensor of shape (n_residues*3, 3)
        rotation_angle: Rotation angle in radians
        rotation_axis: Rotation axis ('x', 'y', or 'z')
        translation: Translation vector [x, y, z]

    Returns:
        Tuple of (transformed_coords, rotation_matrix, translation_vector)
    """
    cos_a, sin_a = math.cos(rotation_angle), math.sin(rotation_angle)

    # Define rotation matrices for different axes
    if rotation_axis == 'x':
        R = torch.tensor([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ], dtype=torch.float32)
    elif rotation_axis == 'y':
        R = torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=torch.float32)
    else:  # z-axis
        R = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

    t = torch.tensor(translation, dtype=torch.float32)

    # Apply transformation to all backbone coordinates: coords_new = R @ coords + t
    coords_transformed = (R @ coords.T).T + t

    return coords_transformed, R, t


def benchmark_single_pdb(pdb_file: str,
                         rotation_angles: List[float] = [math.pi / 6, math.pi / 4, math.pi / 3],
                         rotation_axes: List[str] = ['x', 'y', 'z'],
                         translations: List[List[float]] = [[1.0, 2.0, 0.5], [2.0, 3.0, 1.0],
                                                            [-1.5, 1.0, -2.0]]) -> Dict:
    """
    Benchmark Kabsch alignment on a single PDB file with multiple transformations.

    Args:
        pdb_file: Path to PDB file
        rotation_angles: List of rotation angles to test
        rotation_axes: List of rotation axes to test
        translations: List of translation vectors to test

    Returns:
        Dictionary with benchmark results
    """
    # Parse PDB file
    original_coords = parse_pdb_backbone(pdb_file)
    if original_coords is None:
        return {'error': 'Failed to parse PDB file or insufficient backbone atoms'}

    results = {
        'pdb_file': os.path.basename(pdb_file),
        'n_backbone_atoms': original_coords.shape[0],
        'n_residues': original_coords.shape[0] // 3,
        'transformations': [],
        'rotation_errors': [],
        'translation_errors': [],
        'rmsd_errors': [],
        'avg_rotation_error': 0.0,
        'avg_translation_error': 0.0,
        'avg_rmsd_error': 0.0
    }

    transformation_count = 0
    total_rotation_error = 0.0
    total_translation_error = 0.0
    total_rmsd_error = 0.0

    # Test different combinations of transformations
    for angle in rotation_angles:
        for axis in rotation_axes:
            for translation in translations:
                try:
                    # Apply known transformation
                    transformed_coords, known_R, known_t = apply_known_transformation(
                        original_coords, angle, axis, translation
                    )

                    # Perform Kabsch alignment to recover the transformation
                    # Since we're now using CoordTensor (all 3 backbone atoms), set ca_only=False
                    recovered_R, recovered_t = kabsch(
                        transformed_coords, original_coords,
                        ca_only=False,  # Use all backbone atoms for alignment
                        allow_reflections=False,
                        return_transformed=False
                    )

                    # Calculate expected inverse transformation
                    expected_R_inv = known_R.T
                    expected_t_inv = -expected_R_inv @ known_t

                    # Calculate errors
                    rotation_error = torch.norm(recovered_R - expected_R_inv).item()
                    translation_error = torch.norm(recovered_t - expected_t_inv).item()

                    # Calculate RMSD after alignment using all backbone atoms
                    aligned_coords = kabsch(
                        transformed_coords, original_coords,
                        ca_only=False,  # Use all backbone atoms for alignment
                        allow_reflections=False,
                        return_transformed=True
                    )

                    # Calculate RMSD using all backbone atoms
                    rmsd_error = torch.sqrt(torch.mean((aligned_coords - original_coords) ** 2)).item()

                    # Store results
                    transformation_info = {
                        'angle_deg': angle * 180 / math.pi,
                        'axis': axis,
                        'translation': translation,
                        'rotation_error': rotation_error,
                        'translation_error': translation_error,
                        'rmsd_error': rmsd_error,
                        'det_recovered': torch.det(recovered_R).item()
                    }

                    results['transformations'].append(transformation_info)
                    results['rotation_errors'].append(rotation_error)
                    results['translation_errors'].append(translation_error)
                    results['rmsd_errors'].append(rmsd_error)

                    total_rotation_error += rotation_error
                    total_translation_error += translation_error
                    total_rmsd_error += rmsd_error
                    transformation_count += 1

                except Exception as e:
                    print(f"Error processing transformation for {pdb_file}: {e}")
                    continue

    # Calculate averages
    if transformation_count > 0:
        results['avg_rotation_error'] = total_rotation_error / transformation_count
        results['avg_translation_error'] = total_translation_error / transformation_count
        results['avg_rmsd_error'] = total_rmsd_error / transformation_count

    return results


def benchmark_pdb_directory(pdb_directory: str,
                            max_files: Optional[int] = None,
                            verbose: bool = True) -> Dict:
    """
    Benchmark Kabsch alignment on all PDB files in a directory.

    Args:
        pdb_directory: Directory containing PDB files
        max_files: Maximum number of files to process (None for all)
        verbose: Whether to print progress information

    Returns:
        Dictionary with overall benchmark results
    """
    # Find all PDB files
    pdb_pattern = os.path.join(pdb_directory, "*.pdb")
    pdb_files = glob.glob(pdb_pattern)

    if not pdb_files:
        return {'error': f'No PDB files found in {pdb_directory}'}

    if max_files:
        pdb_files = pdb_files[:max_files]

    if verbose:
        print(f"Found {len(pdb_files)} PDB files to process")

    overall_results = {
        'directory': pdb_directory,
        'total_files': len(pdb_files),
        'processed_files': 0,
        'failed_files': 0,
        'file_results': [],
        'overall_avg_rotation_error': 0.0,
        'overall_avg_translation_error': 0.0,
        'overall_avg_rmsd_error': 0.0,
        'rotation_error_std': 0.0,
        'translation_error_std': 0.0,
        'rmsd_error_std': 0.0
    }

    all_rotation_errors = []
    all_translation_errors = []
    all_rmsd_errors = []
    processed_count = 0

    # Use tqdm for progress bar
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files", disable=not verbose):
        try:
            file_results = benchmark_single_pdb(pdb_file)

            if 'error' in file_results:
                overall_results['failed_files'] += 1
                continue

            overall_results['file_results'].append(file_results)
            overall_results['processed_files'] += 1
            processed_count += 1

            # Collect errors for overall statistics
            all_rotation_errors.extend(file_results['rotation_errors'])
            all_translation_errors.extend(file_results['translation_errors'])
            all_rmsd_errors.extend(file_results['rmsd_errors'])

        except Exception as e:
            overall_results['failed_files'] += 1
            continue

    # Calculate overall statistics
    if all_rotation_errors:
        overall_results['overall_avg_rotation_error'] = np.mean(all_rotation_errors)
        overall_results['overall_avg_translation_error'] = np.mean(all_translation_errors)
        overall_results['overall_avg_rmsd_error'] = np.mean(all_rmsd_errors)

        overall_results['rotation_error_std'] = np.std(all_rotation_errors)
        overall_results['translation_error_std'] = np.std(all_translation_errors)
        overall_results['rmsd_error_std'] = np.std(all_rmsd_errors)

    return overall_results


def print_benchmark_summary(results: Dict):
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 80)
    print("🧪 KABSCH ALIGNMENT BENCHMARK RESULTS")
    print("=" * 80)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return

    print(f"📁 Directory: {results['directory']}")
    print(f"📊 Files processed: {results['processed_files']}/{results['total_files']}")
    print(f"❌ Failed files: {results['failed_files']}")

    if results['processed_files'] == 0:
        print("No files were successfully processed.")
        return

    print(f"\n📈 OVERALL ACCURACY METRICS:")
    print(
        f"   Average Rotation Error:    {results['overall_avg_rotation_error']:.8f} ± {results['rotation_error_std']:.8f}")
    print(
        f"   Average Translation Error: {results['overall_avg_translation_error']:.8f} ± {results['translation_error_std']:.8f}")
    print(f"   Average RMSD Error:        {results['overall_avg_rmsd_error']:.8f} ± {results['rmsd_error_std']:.8f}")

    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    excellent_rmsd = sum(1 for fr in results['file_results'] if fr['avg_rmsd_error'] < 1e-6)
    good_rmsd = sum(1 for fr in results['file_results'] if 1e-6 <= fr['avg_rmsd_error'] < 1e-5)
    acceptable_rmsd = sum(1 for fr in results['file_results'] if 1e-5 <= fr['avg_rmsd_error'] < 1e-4)
    poor_rmsd = sum(1 for fr in results['file_results'] if fr['avg_rmsd_error'] >= 1e-4)

    print(f"   Excellent (RMSD < 1e-6):   {excellent_rmsd} files")
    print(f"   Good (1e-6 ≤ RMSD < 1e-5): {good_rmsd} files")
    print(f"   Acceptable (1e-5 ≤ RMSD < 1e-4): {acceptable_rmsd} files")
    print(f"   Poor (RMSD ≥ 1e-4):        {poor_rmsd} files")

    print("=" * 80)


if __name__ == "__main__":

    PDB_DIRECTORY = "/mnt/hdd8/mehdi/datasets/vqvae/cameo"

    # Optional: Limit number of files for testing
    MAX_FILES = 600  # Set to None to process all files

    print("🧪 Starting Kabsch Alignment Benchmark on PDB Files")
    print(f"📁 Target directory: {PDB_DIRECTORY}")

    # Check if directory exists
    if not os.path.exists(PDB_DIRECTORY):
        print(f"❌ Error: Directory {PDB_DIRECTORY} does not exist!")
        print("Please update the PDB_DIRECTORY variable in the main block.")
        exit(1)

    # Run benchmark
    results = benchmark_pdb_directory(
        pdb_directory=PDB_DIRECTORY,
        max_files=MAX_FILES,
        verbose=True
    )

    # Print results
    print_benchmark_summary(results)

    # Optional: Save results to file
    import json

    output_file = f"kabsch_benchmark_results_{Path(PDB_DIRECTORY).name}.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, np.number) else x))
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
