"""
Sequence length vs RMSD scatter plot with regression line (PDF output).

This manuscript plotting script replicates the visual style shown in the
attached example figure: a dense scatter of RMSD as a function of sequence
length, a red least‑squares regression line, dashed grid lines, and a text
panel reporting Pearson r, Spearman ρ, and the regression slope.

Notes
- Parameters are hardcoded (no CLI) for reproducibility and simplicity.
- Input CSV is expected to contain at least the following columns:
  - num_amino_acids: integer sequence length per sample
  - rmsd: backbone RMSD value for the sample
- Output is a vector PDF saved under the plots/ directory.

Data
- CSV: /mnt/hdd8/mehdi/projects/vq_encoder_decoder/data/cluster_maximum_similarity_50_merged.csv
- PDF: /mnt/hdd8/mehdi/projects/vq_encoder_decoder/plots/seq_len_vs_rmsd_cluster50.pdf
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Hardcoded paths (project-local absolute paths as requested)
CSV_PATH = \
    "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/data/merged_final_results.csv"
OUT_PDF_PATH = \
    "/mnt/hdd8/mehdi/projects/vq_encoder_decoder/plots/seq_len_vs_rmsd_cluster50.pdf"


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a pandas Series to numeric, keeping NaN for invalid values."""
    return pd.to_numeric(series, errors="coerce")


def _compute_metrics(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (pearson_r, spearman_rho, slope, intercept) for y ~ slope*x + intercept.

    Uses numpy/pandas implementations to avoid extra dependencies.
    """
    # Pearson and Spearman using pandas for robustness
    s_x = pd.Series(x)
    s_y = pd.Series(y)
    pearson_r = float(s_x.corr(s_y, method="pearson"))
    spearman_rho = float(s_x.corr(s_y, method="spearman"))

    # Ordinary least squares fit (degree-1 polynomial)
    slope, intercept = np.polyfit(x, y, 1)
    return pearson_r, spearman_rho, float(slope), float(intercept)


def _style_axes(ax: plt.Axes) -> None:
    """Apply the manuscript style (dashed grid, subtle spines)."""
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    # Load and clean data
    df = pd.read_csv(CSV_PATH)
    if "num_amino_acids" not in df.columns or "rmsd" not in df.columns:
        missing = {c for c in ["num_amino_acids", "rmsd"] if c not in df.columns}
        raise KeyError(f"CSV missing required columns: {sorted(missing)}")

    x = _coerce_numeric(df["num_amino_acids"])  # sequence length
    y = _coerce_numeric(df["rmsd"])             # RMSD
    mask = (~x.isna()) & (~y.isna())
    x = x.loc[mask].to_numpy(dtype=float)
    y = y.loc[mask].to_numpy(dtype=float)

    if x.size == 0:
        raise ValueError("No valid rows after removing NaNs in num_amino_acids/rmsd.")

    # Metrics and regression
    pearson_r, spearman_rho, slope, intercept = _compute_metrics(x, y)

    # Prepare regression line across the observed x-range
    xmin, xmax = float(np.min(x)), float(np.max(x))
    x_fit = np.linspace(xmin, xmax, 200)
    y_fit = slope * x_fit + intercept

    # Plot
    plt.figure(figsize=(12, 6.5))
    ax = plt.gca()
    _style_axes(ax)

    # Scatter: small, semi-transparent points
    ax.scatter(x, y, s=22, c="#4C9BD6", alpha=0.55, edgecolors="none")

    # Regression line
    ax.plot(x_fit, y_fit, color="red", linewidth=2.0)

    # Labels
    ax.set_xlabel("Sequence length", fontsize=14)
    ax.set_ylabel("RMSD", fontsize=14)

    # Annotation panel in upper-left corner
    text = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"Spearman ρ = {spearman_rho:.3f}\n"
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

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(OUT_PDF_PATH)
    print(f"Saved figure to: {OUT_PDF_PATH}")


if __name__ == "__main__":
    main()


