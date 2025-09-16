#!/usr/bin/env python3
"""Secondary-structure assignment from backbone-only PDBs.

The script walks directories of PDB files, determines per-file fractions of
alpha-helix (H), beta-sheet (E), and coil (C) using backbone dihedrals, and
writes a single CSV summary. Each CSV row contains seven values:

    pdb_name, percent_alpha, percent_beta, percent_coil,
    count_alpha, count_beta, count_coil

Multiprocessing and a progress bar keep large batches responsive.
"""

import argparse
import csv
import glob
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from tqdm import tqdm


# ---------- geometry helpers ----------
def vec(a, b):
    """Return the 3D vector pointing from coordinate ``a`` to coordinate ``b``.

    Args:
        a: Tuple ``(x, y, z)`` for the starting point.
        b: Tuple ``(x, y, z)`` for the destination point.

    Returns:
        Tuple of floats representing ``b - a`` for each Cartesian axis.
    """
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def norm(v):
    """Compute the Euclidean length of a 3D vector.

    Args:
        v: Tuple ``(x, y, z)`` representing the vector.

    Returns:
        Float length ``sqrt(x^2 + y^2 + z^2)``.
    """
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def cross(a, b):
    """Return the cross product between two 3D vectors.

    Args:
        a: First vector as ``(x, y, z)``.
        b: Second vector as ``(x, y, z)``.

    Returns:
        Tuple representing ``a × b`` component-wise.
    """
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    """Return the dot product between two 3D vectors.

    Args:
        a: First vector as ``(x, y, z)``.
        b: Second vector as ``(x, y, z)``.

    Returns:
        Float representing ``a · b``.
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def dihedral(p1, p2, p3, p4):
    """Return the signed dihedral angle for four backbone atoms.

    Args:
        p1: Coordinate tuple for the first atom (typically C of residue i-1).
        p2: Coordinate tuple for the second atom (typically N of residue i).
        p3: Coordinate tuple for the third atom (typically CA of residue i).
        p4: Coordinate tuple for the fourth atom (typically C of residue i).

    Returns:
        Dihedral angle in degrees following a consistent right-handed
        convention suitable for Ramachandran analysis.
    """

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
    """Check peptide C–N separation to avoid spanning gaps when computing dihedrals.

    Args:
        a: Coordinate tuple for the first atom (carbonyl C).
        b: Coordinate tuple for the second atom (amide N).
        maxdist: Upper bound on bond length in Ångström allowed for a peptide bond.
        mindist: Lower bound on bond length in Ångström allowed for a peptide bond.

    Returns:
        ``True`` if the squared distance between the atoms lies within the
        expected peptide bond limits, otherwise ``False``.
    """

    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    d2 = dx * dx + dy * dy + dz * dz
    return (mindist ** 2) <= d2 <= (maxdist ** 2)


# ---------- PDB parsing ----------
def parse_backbone(pdb_path: str) -> Dict[str, List[Tuple[Tuple[str, int, str], Dict[str, Tuple[float, float, float]], str]]]:
    """Parse backbone atoms from a PDB file grouped by chain.

    Args:
        pdb_path: Path to the PDB file to be read.

    Returns:
        Dictionary mapping chain identifiers to ordered residue entries.
        Each entry is a tuple ``((chain, resseq, icode), atom_dict, resname)``
        where ``atom_dict`` holds the N/CA/C coordinates if present.
    """

    atoms_by_res: Dict[Tuple[str, int, str], Dict[str, Tuple[float, float, float]]] = {}
    resname_by_res: Dict[Tuple[str, int, str], str] = {}
    with open(pdb_path) as f:
        for ln in f:
            if not ln.startswith("ATOM"):
                continue
            name = ln[12:16].strip()
            if name not in ("N", "CA", "C"):
                continue
            resname = ln[17:20].strip()
            chain = ln[21].strip() or "A"
            try:
                resseq = int(ln[22:26])
            except ValueError:
                continue
            icode = ln[26].strip()
            x = float(ln[30:38])
            y = float(ln[38:46])
            z = float(ln[46:54])
            key = (chain, resseq, icode)
            atoms_by_res.setdefault(key, {})[name] = (x, y, z)
            resname_by_res[key] = resname
    keys = sorted(atoms_by_res, key=lambda k: (k[0], k[1], k[2]))
    chains = defaultdict(list)
    for k in keys:
        chains[k[0]].append((k, atoms_by_res[k], resname_by_res[k]))
    return chains


# ---------- phi/psi & assignment ----------
def compute_phi_psi(chains):
    """Compute per-residue phi/psi angles for each chain.

    Args:
        chains: Mapping produced by :func:`parse_backbone` describing residue
            order and backbone coordinates for each chain.

    Returns:
        List of rows ``[chain_id, resseq, icode, resname, phi, psi]`` with
        dihedral angles in degrees or ``None`` when unavailable.
    """
    rows = []
    for chain, items in chains.items():
        n = len(items)
        for i, (key, atoms, resn) in enumerate(items):
            phi = psi = None
            if i > 0 and "C" in items[i - 1][1] and all(x in atoms for x in ("N", "CA", "C")):
                if bond_ok(items[i - 1][1]["C"], atoms["N"]):
                    phi = dihedral(items[i - 1][1]["C"], atoms["N"], atoms["CA"], atoms["C"])
            if i < n - 1 and "N" in items[i + 1][1] and all(x in atoms for x in ("N", "CA", "C")):
                if bond_ok(atoms["C"], items[i + 1][1]["N"]):
                    psi = dihedral(atoms["N"], atoms["CA"], atoms["C"], items[i + 1][1]["N"])
            rows.append([chain, key[1], key[2], resn, phi, psi])
    return rows


def assign(phi, psi):
    """Assign secondary-structure class from phi/psi dihedral angles.

    Args:
        phi: Phi dihedral angle in degrees or ``None`` if not computable.
        psi: Psi dihedral angle in degrees or ``None`` if not computable.

    Returns:
        One of ``"H"`` (alpha helix), ``"E"`` (beta sheet), or ``"C"`` (coil).
    """
    if phi is None or psi is None:
        return "C"
    if -90 <= phi <= -30 and -70 <= psi <= -15:
        return "H"
    if -160 <= phi <= -80 and 90 <= psi <= 180:
        return "E"
    return "C"


def smooth(labels, minH=4, minE=3):
    """Post-process raw secondary-structure labels to suppress short runs.

    Args:
        labels: List of raw assignments containing ``H``/``E``/``C``.
        minH: Minimum contiguous length required to keep a helix designation.
        minE: Minimum contiguous length required to keep a beta-strand designation.

    Returns:
        New list of labels where short helix/strand runs are converted to coil.
    """
    lab = labels[:]
    n = len(lab)
    i = 0
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        current_label = lab[i]
        run = j - i
        required = minH if current_label == "H" else (minE if current_label == "E" else 1)
        if run < required:
            for k in range(i, j):
                lab[k] = "C"
        i = j
    return lab


# ---------- helpers for file discovery & summarising ----------
def find_pdb_files(directory_path: str) -> List[str]:
    """Return all PDB files under ``directory_path`` recursively."""
    pattern = os.path.join(directory_path, "**", "*.pdb")
    return glob.glob(pattern, recursive=True)


def gather_pdb_paths(inputs: List[str]) -> List[str]:
    """Collect a sorted list of unique PDB file paths from inputs.

    Args:
        inputs: Iterable of file or directory paths.

    Returns:
        Sorted list of absolute or relative paths pointing to PDB files.
    """
    files: List[str] = []
    for path in inputs:
        if os.path.isdir(path):
            files.extend(find_pdb_files(path))
        elif path.lower().endswith(".pdb") and os.path.isfile(path):
            files.append(path)
    # Remove duplicates while keeping deterministic order
    return sorted(set(files))


def summarise_structure(pdb_path: str) -> Dict[str, float]:
    """Summarise secondary structure fractions for a single PDB file.

    Args:
        pdb_path: Path to the PDB file being analysed.

    Returns:
        Dictionary with the file name, raw counts of ``H``/``E``/``C`` residues,
        and their percentages relative to the total residues examined.
    """
    chains = parse_backbone(pdb_path)
    counts = Counter()
    for ch, items in chains.items():
        rows = compute_phi_psi({ch: items})
        if not rows:
            continue
        raw = [assign(r[4], r[5]) for r in rows]
        labels = smooth(raw, minH=4, minE=3)
        counts.update(labels)

    total = sum(counts.values())
    alpha = counts.get("H", 0)
    beta = counts.get("E", 0)
    coil = total - alpha - beta
    if coil < 0:
        coil = counts.get("C", 0)

    if total > 0:
        percent_alpha = (alpha / total) * 100.0
        percent_beta = (beta / total) * 100.0
        percent_coil = (coil / total) * 100.0
    else:
        percent_alpha = percent_beta = percent_coil = 0.0

    return {
        "pdb_name": os.path.basename(pdb_path),
        "percent_alpha": percent_alpha,
        "percent_beta": percent_beta,
        "percent_coil": percent_coil,
        "count_alpha": alpha,
        "count_beta": beta,
        "count_coil": coil,
    }


def process_file_task(pdb_path: str):
    """Wrapper for multiprocessing that records success or failure.

    Args:
        pdb_path: Path to the PDB file being processed.

    Returns:
        Tuple ``(True, summary_dict)`` on success or ``(False, (path, message))``
        on failure.
    """
    try:
        summary = summarise_structure(pdb_path)
        return True, summary
    except Exception as exc:
        return False, (pdb_path, str(exc))


def write_summary_csv(records: List[Dict[str, float]], output_path: str):
    """Write aggregated secondary-structure statistics to CSV.

    Args:
        records: Sequence of dictionaries returned by :func:`summarise_structure`.
        output_path: Destination path where the CSV file will be written.
    """
    fieldnames = [
        "pdb_name",
        "percent_alpha",
        "percent_beta",
        "percent_coil",
        "count_alpha",
        "count_beta",
        "count_coil",
    ]
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main():
    """Parse CLI arguments, run batch processing, and emit a summary CSV."""
    parser = argparse.ArgumentParser(
        description="Assign secondary structure from backbone-only PDB files and summarise results.")
    parser.add_argument(
        "--data",
        type=str,
        nargs='+',
        required=True,
        help="Paths to PDB files or directories containing PDBs.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="secondary_structure_summary.csv",
        help="Destination CSV file for aggregated results.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes to use (default: CPU count).",
    )
    args = parser.parse_args()

    pdb_files = gather_pdb_paths(args.data)
    if not pdb_files:
        print("No PDB files found for the provided inputs.")
        return

    results = []
    errors = []
    max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else 1

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file_task, path): path for path in pdb_files}
            for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Assigning secondary structure"):
                success, payload = future.result()
                if success:
                    results.append(payload)
                else:
                    errors.append(payload)
    else:
        for path in tqdm(pdb_files, desc="Assigning secondary structure"):
            success, payload = process_file_task(path)
            if success:
                results.append(payload)
            else:
                errors.append(payload)

    if results:
        results.sort(key=lambda r: r["pdb_name"])
        write_summary_csv(results, args.output_csv)
        print(f"Wrote summary for {len(results)} structures to {args.output_csv}")
    else:
        print("No structures processed successfully; summary file not generated.")

    if errors:
        print(f"Encountered {len(errors)} errors during processing:")
        for path, message in errors:
            print(f"  {path}: {message}")


if __name__ == "__main__":
    main()
