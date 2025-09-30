#!/usr/bin/env python3
"""Compute NaN statistics for backbone H5 files.

This script scans a directory of HDF5 files produced by ``pdb_to_h5.py`` and
reports, for each file, how many residues have missing backbone atoms (NaN) as well as
what fraction of residues are missing backbone atoms.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import csv


def nan_metrics(h5_path: Path):
    """Return (nan_residue_count, missing_residue_ratio, residue_count)."""
    with h5py.File(h5_path, "r") as handle:
        coords = handle["N_CA_C_O_coord"][...]

    total_residues = int(coords.shape[0]) if coords.ndim >= 1 else 0
    if total_residues == 0:
        return 0, 0.0, 0

    # Identify residues where any backbone coordinate is NaN
    residue_nan_mask = np.isnan(coords).any(axis=(1, 2))
    nan_residue_count = int(residue_nan_mask.sum())
    missing_residue_ratio = float(nan_residue_count / total_residues)

    return nan_residue_count, missing_residue_ratio, total_residues


def process_directory(h5_dir: Path):
    """Yield per-file metric dictionaries sorted by file name."""
    files = sorted(h5_dir.glob("*.h5"))
    for path in files:
        nan_residue_count, missing_ratio, total_residues = nan_metrics(path)
        yield {
            "pdb_name": path.stem,
            "nan_residue_count": nan_residue_count,
            "missing_residue_ratio": missing_ratio,
            "total_residues": total_residues,
        }


def main():
    parser = argparse.ArgumentParser(description="Compute NaN statistics for H5 files.")
    parser.add_argument("--h5_dir", required=True, type=Path, help="Directory containing .h5 files")
    parser.add_argument("--output", required=True, type=Path, help="Path to output CSV")
    args = parser.parse_args()

    records = list(process_directory(args.h5_dir))

    if not records:
        print(f"No .h5 files found in {args.h5_dir}")
        return

    fieldnames = ["pdb_name", "nan_residue_count", "missing_residue_ratio", "total_residues"]
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"Wrote NaN metrics for {len(records)} files to {args.output}")


if __name__ == "__main__":
    main()
