#!/usr/bin/env python3
"""Ramachandran plot analysis colored by VQ codebook tokens.

This script analyzes the relationship between VQ codes and backbone φ/ψ angles
by creating Ramachandran plots where each point is colored by its assigned VQ code.

Usage:
    python -m codebook_evaluation.ramachandran_by_vq_code \
        --eval_dir evaluation_results/2025-11-24__19-21-20 \
        --top_k 10
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ---------- Geometry helpers (from assign_ss_from_backbone.py) ----------
def vec(a, b):
    """Return the 3D vector pointing from coordinate a to coordinate b."""
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def norm(v):
    """Compute the Euclidean length of a 3D vector."""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def cross(a, b):
    """Return the cross product between two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    """Return the dot product between two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def dihedral(p1, p2, p3, p4):
    """Return the signed dihedral angle for four backbone atoms in degrees."""
    b0 = vec(p2, p1)
    b1 = vec(p2, p3)
    b2 = vec(p3, p4)
    b1_len = norm(b1) + 1e-9
    b1n = (b1[0] / b1_len, b1[1] / b1_len, b1[2] / b1_len)
    v = cross(b0, b1n)
    w = cross(b2, b1n)
    x = dot(v, w)
    y = dot(cross(v, w), b1n)
    return math.degrees(math.atan2(y, x))


def bond_ok(a, b, maxdist=1.8, mindist=1.1):
    """Check peptide C–N separation to avoid spanning gaps."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    d2 = dx * dx + dy * dy + dz * dz
    return (mindist ** 2) <= d2 <= (maxdist ** 2)


# ---------- PDB parsing ----------
def parse_backbone(pdb_path: str) -> List[Tuple[int, Dict[str, Tuple[float, float, float]]]]:
    """Parse backbone atoms from a PDB file, returning list of (resseq, atom_dict).
    
    Returns:
        List of tuples (residue_index, {atom_name: (x, y, z)}) sorted by residue number.
    """
    atoms_by_res: Dict[int, Dict[str, Tuple[float, float, float]]] = {}
    
    with open(pdb_path) as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            name = ln[12:16].strip()
            if name not in ("N", "CA", "C"):
                continue
            try:
                resseq = int(ln[22:26])
            except ValueError:
                continue
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
            atoms_by_res.setdefault(resseq, {})[name] = (x, y, z)
    
    # Sort by residue number and return as list
    sorted_res = sorted(atoms_by_res.keys())
    return [(r, atoms_by_res[r]) for r in sorted_res]


def compute_phi_psi_list(residues: List[Tuple[int, Dict[str, Tuple[float, float, float]]]]) -> List[Tuple[int, Optional[float], Optional[float]]]:
    """Compute per-residue phi/psi angles.
    
    Args:
        residues: List of (resseq, atom_dict) from parse_backbone.
        
    Returns:
        List of (residue_index, phi, psi) where phi/psi are in degrees or None.
    """
    results = []
    n = len(residues)
    
    for i, (resseq, atoms) in enumerate(residues):
        phi = None
        psi = None
        
        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0 and "C" in residues[i - 1][1] and all(x in atoms for x in ("N", "CA", "C")):
            prev_c = residues[i - 1][1]["C"]
            if bond_ok(prev_c, atoms["N"]):
                phi = dihedral(prev_c, atoms["N"], atoms["CA"], atoms["C"])
        
        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        if i < n - 1 and "N" in residues[i + 1][1] and all(x in atoms for x in ("N", "CA", "C")):
            next_n = residues[i + 1][1]["N"]
            if bond_ok(atoms["C"], next_n):
                psi = dihedral(atoms["N"], atoms["CA"], atoms["C"], next_n)
        
        results.append((resseq, phi, psi))
    
    return results


# ---------- Data loading ----------
def load_vq_indices(csv_path: str) -> Dict[str, List[int]]:
    """Load VQ indices from CSV file.
    
    Returns:
        Dict mapping pid -> list of VQ codes per residue.
    """
    indices_dict = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['pid']
            indices_str = row['indices'].strip()
            if indices_str:
                indices = [int(x) for x in indices_str.split()]
                indices_dict[pid] = indices
    return indices_dict


def process_single_protein(args: Tuple[str, str, List[int]]) -> List[Tuple[int, float, float]]:
    """Process a single protein to extract (vq_code, phi, psi) tuples.
    
    Args:
        args: Tuple of (pid, pdb_path, vq_codes)
        
    Returns:
        List of (vq_code, phi, psi) for residues with valid angles.
    """
    pid, pdb_path, vq_codes = args
    
    if not os.path.exists(pdb_path):
        return []
    
    try:
        residues = parse_backbone(pdb_path)
        phi_psi_list = compute_phi_psi_list(residues)
        
        results = []
        # Match by position (0-indexed)
        for i, (resseq, phi, psi) in enumerate(phi_psi_list):
            if i >= len(vq_codes):
                break
            if phi is not None and psi is not None:
                vq_code = vq_codes[i]
                if vq_code >= 0:  # Skip padding tokens
                    results.append((vq_code, phi, psi))
        
        return results
    except Exception as e:
        return []


# ---------- Visualization ----------
def get_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """Generate n visually distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i / n
        # Use high saturation and value for visibility
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(rgb)
    return colors


def assign_ss_label(phi: float, psi: float) -> str:
    """Assign secondary structure label based on phi/psi angles."""
    if -90 <= phi <= -30 and -70 <= psi <= -15:
        return "α-helix"
    elif -160 <= phi <= -80 and 90 <= psi <= 180:
        return "β-sheet"
    else:
        return "coil"


def plot_ramachandran_by_code(
    data: List[Tuple[int, float, float]],
    top_k: int,
    output_path: str,
    title: str = "Ramachandran Plot by VQ Code"
):
    """Create Ramachandran plot colored by VQ code.
    
    Args:
        data: List of (vq_code, phi, psi) tuples.
        top_k: Number of top codes to highlight (rest grouped as "other").
        output_path: Path to save the plot.
        title: Plot title.
    """
    # Count frequency of each code
    code_counts = defaultdict(int)
    for code, _, _ in data:
        code_counts[code] += 1
    
    # Get top-K codes
    sorted_codes = sorted(code_counts.items(), key=lambda x: -x[1])
    top_codes = [code for code, _ in sorted_codes[:top_k]]
    top_code_set = set(top_codes)
    
    # Assign colors
    colors = get_distinct_colors(top_k)
    code_to_color = {code: colors[i] for i, code in enumerate(top_codes)}
    other_color = (0.7, 0.7, 0.7)  # Gray for "other"
    
    # Separate data
    other_phi, other_psi = [], []
    code_data = {code: ([], []) for code in top_codes}
    
    for code, phi, psi in data:
        if code in top_code_set:
            code_data[code][0].append(phi)
            code_data[code][1].append(psi)
        else:
            other_phi.append(phi)
            other_psi.append(psi)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot "other" first (background)
    if other_phi:
        ax.scatter(other_phi, other_psi, c=[other_color], s=3, alpha=0.3, label='Other codes')
    
    # Plot top-K codes
    for code in reversed(top_codes):  # Plot least frequent on top
        phi_vals, psi_vals = code_data[code]
        if phi_vals:
            count = code_counts[code]
            pct = 100.0 * count / len(data)
            ax.scatter(
                phi_vals, psi_vals,
                c=[code_to_color[code]],
                s=8,
                alpha=0.6,
                label=f'Code {code} ({pct:.2f}%)'
            )
    
    # Add reference regions
    # Alpha-helix region
    alpha_rect = plt.Rectangle((-90, -70), 60, 55, fill=False, 
                                edgecolor='blue', linestyle='--', linewidth=3, label='α-helix region')
    ax.add_patch(alpha_rect)
    
    # Beta-sheet region
    beta_rect = plt.Rectangle((-160, 90), 80, 90, fill=False,
                               edgecolor='purple', linestyle='--', linewidth=3, label='β-sheet region')
    ax.add_patch(beta_rect)
    
    # Formatting
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel('φ (phi) [degrees]', fontsize=12)
    ax.set_ylabel('ψ (psi) [degrees]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='upper right', fontsize=8, markerscale=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved Ramachandran plot to: {output_path}")


def plot_per_code_distributions(
    data: List[Tuple[int, float, float]],
    top_k: int,
    output_dir: str
):
    """Create individual Ramachandran plots for each top-K code.
    
    Args:
        data: List of (vq_code, phi, psi) tuples.
        top_k: Number of top codes to plot.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Count frequency of each code
    code_counts = defaultdict(int)
    for code, _, _ in data:
        code_counts[code] += 1
    
    sorted_codes = sorted(code_counts.items(), key=lambda x: -x[1])
    top_codes = [code for code, _ in sorted_codes[:top_k]]
    
    # Group data by code
    code_data = defaultdict(list)
    for code, phi, psi in data:
        code_data[code].append((phi, psi))
    
    for rank, code in enumerate(top_codes, 1):
        points = code_data[code]
        if not points:
            continue
        
        phi_vals = [p[0] for p in points]
        psi_vals = [p[1] for p in points]
        
        # Compute statistics
        mean_phi = np.mean(phi_vals)
        mean_psi = np.mean(psi_vals)
        std_phi = np.std(phi_vals)
        std_psi = np.std(psi_vals)
        
        # Assign predominant SS
        ss_counts = defaultdict(int)
        for phi, psi in points:
            ss = assign_ss_label(phi, psi)
            ss_counts[ss] += 1
        predominant_ss = max(ss_counts.items(), key=lambda x: x[1])[0]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter with density coloring
        ax.scatter(phi_vals, psi_vals, s=5, alpha=0.5, c='steelblue')
        
        # Mark mean
        ax.scatter([mean_phi], [mean_psi], s=200, c='red', marker='x', linewidths=3, 
                   label=f'Mean: ({mean_phi:.1f}°, {mean_psi:.1f}°)')
        
        # Add reference regions
        alpha_rect = plt.Rectangle((-90, -70), 60, 55, fill=False, 
                                    edgecolor='blue', linestyle='--', linewidth=3)
        ax.add_patch(alpha_rect)
        beta_rect = plt.Rectangle((-160, 90), 80, 90, fill=False,
                                   edgecolor='purple', linestyle='--', linewidth=3)
        ax.add_patch(beta_rect)
        
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel('φ (phi) [degrees]', fontsize=12)
        ax.set_ylabel('ψ (psi) [degrees]', fontsize=12)
        
        count = len(points)
        pct = 100.0 * count / len(data)
        ax.set_title(
            f'VQ Code {code} (Rank #{rank})\n'
            f'N={count} ({pct:.2f}%), Predominant: {predominant_ss}\n'
            f'φ: {mean_phi:.1f}° ± {std_phi:.1f}°, ψ: {mean_psi:.1f}° ± {std_psi:.1f}°',
            fontsize=11
        )
        
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"code_{code}_rank{rank}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
    
    print(f"Saved {len(top_codes)} per-code plots to: {output_dir}")


def generate_statistics_report(
    data: List[Tuple[int, float, float]],
    top_k: int,
    output_path: str
):
    """Generate a statistics report for VQ codes and their φ/ψ distributions.
    
    Args:
        data: List of (vq_code, phi, psi) tuples.
        top_k: Number of top codes to report.
        output_path: Path to save the report.
    """
    # Count and group
    code_counts = defaultdict(int)
    code_data = defaultdict(list)
    for code, phi, psi in data:
        code_counts[code] += 1
        code_data[code].append((phi, psi))
    
    sorted_codes = sorted(code_counts.items(), key=lambda x: -x[1])
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VQ Code Ramachandran Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total residues analyzed: {len(data)}\n")
        f.write(f"Unique VQ codes used: {len(code_counts)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"Top-{top_k} VQ Codes by Frequency\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"{'Rank':<6}{'Code':<8}{'Count':<10}{'%':<8}{'Mean φ':<12}{'Mean ψ':<12}{'Std φ':<10}{'Std ψ':<10}{'Predominant SS':<15}\n")
        f.write("-" * 95 + "\n")
        
        for rank, (code, count) in enumerate(sorted_codes[:top_k], 1):
            points = code_data[code]
            phi_vals = [p[0] for p in points]
            psi_vals = [p[1] for p in points]
            
            mean_phi = np.mean(phi_vals)
            mean_psi = np.mean(psi_vals)
            std_phi = np.std(phi_vals)
            std_psi = np.std(psi_vals)
            
            ss_counts = defaultdict(int)
            for phi, psi in points:
                ss = assign_ss_label(phi, psi)
                ss_counts[ss] += 1
            predominant_ss = max(ss_counts.items(), key=lambda x: x[1])[0]
            
            pct = 100.0 * count / len(data)
            f.write(f"{rank:<6}{code:<8}{count:<10}{pct:<8.2f}{mean_phi:<12.1f}{mean_psi:<12.1f}{std_phi:<10.1f}{std_psi:<10.1f}{predominant_ss:<15}\n")
        
        # Secondary structure breakdown
        f.write("\n" + "-" * 80 + "\n")
        f.write("Secondary Structure Distribution by Top Codes\n")
        f.write("-" * 80 + "\n\n")
        
        for rank, (code, count) in enumerate(sorted_codes[:top_k], 1):
            points = code_data[code]
            ss_counts = defaultdict(int)
            for phi, psi in points:
                ss = assign_ss_label(phi, psi)
                ss_counts[ss] += 1
            
            f.write(f"Code {code} (Rank #{rank}):\n")
            for ss in ["α-helix", "β-sheet", "coil"]:
                ss_count = ss_counts.get(ss, 0)
                ss_pct = 100.0 * ss_count / count if count > 0 else 0
                f.write(f"  {ss}: {ss_count} ({ss_pct:.1f}%)\n")
            f.write("\n")
    
    print(f"Saved statistics report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VQ codes vs Ramachandran angles"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Path to evaluation results directory containing vq_indices.csv and PDB folders"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top VQ codes to highlight (default: 10)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: eval_dir/ramachandran_analysis)"
    )
    args = parser.parse_args()
    
    # Always use original PDB files (ground truth φ/ψ angles)
    pdb_folder = "original_pdb_files"
    
    # Paths
    csv_path = os.path.join(args.eval_dir, "vq_indices.csv")
    pdb_dir = os.path.join(args.eval_dir, pdb_folder)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.eval_dir, "ramachandran_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate paths
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"VQ indices CSV not found: {csv_path}")
    if not os.path.isdir(pdb_dir):
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")
    
    print(f"Loading VQ indices from: {csv_path}")
    vq_indices = load_vq_indices(csv_path)
    print(f"Loaded {len(vq_indices)} proteins")
    
    print(f"Using original PDB files from: {pdb_dir}")
    
    # Prepare tasks
    tasks = []
    for pid, codes in vq_indices.items():
        pdb_path = os.path.join(pdb_dir, f"{pid}.pdb")
        tasks.append((pid, pdb_path, codes))
    
    # Process proteins
    all_data = []
    max_workers = args.max_workers or os.cpu_count() or 1
    
    print(f"Processing {len(tasks)} proteins with {max_workers} workers...")
    
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_protein, t): t[0] for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting φ/ψ"):
                result = future.result()
                all_data.extend(result)
    else:
        for task in tqdm(tasks, desc="Extracting φ/ψ"):
            result = process_single_protein(task)
            all_data.extend(result)
    
    print(f"Collected {len(all_data)} (code, φ, ψ) data points")
    
    if not all_data:
        print("No data collected. Check PDB files and VQ indices alignment.")
        return
    
    # Generate outputs
    print("\nGenerating visualizations and reports...")
    
    # 1. Combined Ramachandran plot
    combined_plot_path = os.path.join(output_dir, f"ramachandran_top{args.top_k}.png")
    plot_ramachandran_by_code(
        all_data,
        top_k=args.top_k,
        output_path=combined_plot_path,
        title=f"Ramachandran Plot by Top-{args.top_k} VQ Codes"
    )
    
    # 2. Per-code plots
    per_code_dir = os.path.join(output_dir, "per_code_plots")
    plot_per_code_distributions(all_data, top_k=args.top_k, output_dir=per_code_dir)
    
    # 3. Statistics report
    report_path = os.path.join(output_dir, "statistics_report.txt")
    generate_statistics_report(all_data, top_k=args.top_k, output_path=report_path)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

