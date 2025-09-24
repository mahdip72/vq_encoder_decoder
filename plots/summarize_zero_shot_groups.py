"""
Summarize zero-shot evaluation metrics by structural composition groups.

This script:
- Loads the merged zero-shot CSV using pandas
- Applies the provided grouping rules to form structure groups
- Computes the average RMSD and TM-score for each group
- Prints the results as a simple table to stdout

Notes:
- Parameters are hardcoded as requested (no CLI args)
- Uses the following grouping rules:
  D (coil-rich): percent_coil >= 60
  A (mainly alpha): percent_alpha >= 45 and (percent_alpha - percent_beta) >= 15
  B (mainly beta): percent_beta >= 25 and (percent_beta - percent_alpha) >= 10
  AB (balanced): percent_alpha >= 25 and percent_beta >= 15 and |percent_alpha - percent_beta| <= 10
- "Samples that don't have NaN residues" are rows with nan_value_count == 0

Expected CSV columns (from the provided dataset):
- percent_alpha, percent_beta, percent_coil, tm_score, rmsd, nan_value_count
"""

from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd


CSV_PATH = \
    "/mnt/hdd8/mehdi/datasets/vqvae/zero-shot-set/cluster_maximum_similarity_50_merged.csv"


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_group_means(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric dtypes
    df = _coerce_numeric(
        df,
        [
            "percent_alpha",
            "percent_beta",
            "percent_coil",
            "tm_score",
            "rmsd",
            "nan_value_count",
        ],
    )

    # Convenience aliases
    a = df["percent_alpha"]
    b = df["percent_beta"]
    c = df["percent_coil"]

    # Group masks based on the rules
    mask_all = pd.Series([True] * len(df), index=df.index)
    mask_A = (a >= 45) & ((a - b) >= 15)
    mask_B = (b >= 25) & ((b - a) >= 10)
    mask_AB = (a >= 25) & (b >= 15) & ((a - b).abs() <= 10)
    mask_D = c >= 60
    mask_no_nan_res = df["nan_value_count"].fillna(0) == 0
    mask_has_nan_res = df["nan_value_count"].fillna(0) > 0

    def summarize(name: str, mask: pd.Series) -> Dict[str, object]:
        subset = df.loc[mask]
        mean_rmsd = subset["rmsd"].mean()
        mean_tm = subset["tm_score"].mean()
        return {
            "group": name,
            "count": int(len(subset)),
            "mean_rmsd": float(mean_rmsd) if not math.isnan(mean_rmsd) else float("nan"),
            "mean_tm_score": float(mean_tm) if not math.isnan(mean_tm) else float("nan"),
        }

    rows: List[Dict[str, object]] = [
        summarize("All rows", mask_all),
        summarize("A (mainly α)", mask_A),
        summarize("B (mainly β)", mask_B),
        summarize("AB (mixed)", mask_AB),
        summarize("D (coil-rich/disordered)", mask_D),
        summarize("No-NaN residues", mask_no_nan_res),
        summarize("Has-NaN residues (>=1)", mask_has_nan_res),
    ]

    result = pd.DataFrame(rows, columns=["group", "count", "mean_rmsd", "mean_tm_score"])  # ensure order
    # A little formatting for readability (4 decimals)
    result["mean_rmsd"] = result["mean_rmsd"].round(4)
    result["mean_tm_score"] = result["mean_tm_score"].round(4)
    return result


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    table = compute_group_means(df)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()


