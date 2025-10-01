"""
Number of missing residues vs RMSD scatter with regression (PDF output).

This script mirrors the style of the manuscript scatter/regression plots:
- Blue semi‑transparent scatter points
- Red least‑squares regression line
- Dashed grid and left/bottom spines only
- Text box with Pearson r, Spearman ρ, and slope

Data columns required from the CSV:
- nan_value_count: integer number of missing residues
- rmsd: backbone RMSD value

Hardcoded paths (no argparse) by design for reproducibility.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = \
    "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/data/merged_final_results.csv"
OUT_PDF_PATH = \
    "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/plots/missing_residues_vs_rmsd_cluster50.pdf"


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _corr_and_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    sx = pd.Series(x)
    sy = pd.Series(y)
    pearson = float(sx.corr(sy, method="pearson"))
    spearman = float(sx.corr(sy, method="spearman"))
    slope, intercept = np.polyfit(x, y, 1)
    return pearson, spearman, float(slope), float(intercept)


def _style(ax: plt.Axes) -> None:
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    if "nan_residue_count" not in df.columns or "rmsd" not in df.columns:
        raise KeyError("CSV must contain 'nan_residue_count' and 'rmsd' columns")

    x = _coerce_numeric(df["nan_residue_count"]).to_numpy(dtype=float)
    y = _coerce_numeric(df["rmsd"]).to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("No valid rows to plot after NaN filtering")

    pearson, spearman, slope, intercept = _corr_and_fit(x, y)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    xfit = np.linspace(xmin, xmax, 200)
    yfit = slope * xfit + intercept

    plt.figure(figsize=(12, 6.5))
    ax = plt.gca()
    _style(ax)

    ax.scatter(x, y, s=22, c="#4C9BD6", alpha=0.55, edgecolors="none")
    ax.plot(xfit, yfit, color="red", linewidth=2.0)

    ax.set_xlabel("Number of missing residues", fontsize=14)
    ax.set_ylabel("RMSD", fontsize=14)

    text = (
        f"Pearson r = {pearson:.3f}\n"
        f"Spearman ρ = {spearman:.3f}\n"
        f"Slope = {slope:.4f}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, edgecolor="none"),
    )

    plt.tight_layout()
    plt.savefig(OUT_PDF_PATH)
    print(f"Saved figure to: {OUT_PDF_PATH}")


if __name__ == "__main__":
    main()


