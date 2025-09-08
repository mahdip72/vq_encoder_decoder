#!/usr/bin/env python3
# Compute code perplexity (unigram perplexity) from a CSV of VQ indices.
# CSV columns expected: pid, indices, protein_sequence
# 'indices' must be space-separated token IDs (as written by your script).

import csv
import math
from collections import Counter

# === Configure this ===
CSV_PATH = "path/to/your/result_dir/vq_indices.csv"
SKIP_TOKENS = None  # e.g., set to { -1 } if you want to ignore padding codes

def code_perplexity_from_counts(counts):
    """
    Compute code perplexity from a Counter {code: count}.
    Returns:
        ppl_e: perplexity using natural log base (e)
        ppl_2: perplexity using log base 2 (bits)
        H_e: entropy in nats
        H_2: entropy in bits
    """
    N = sum(counts.values())
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Entropy in nats
    H_e = 0.0
    for c in counts.values():
        p = c / N
        # p > 0 always since counts>0
        H_e -= p * math.log(p)

    # Convert to bits
    H_2 = H_e / math.log(2.0)

    ppl_e = math.exp(H_e)
    ppl_2 = 2.0 ** H_2
    return ppl_e, ppl_2, H_e, H_2

def main():
    counts = Counter()
    total_rows = 0
    total_tokens = 0

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            idx_str = row.get("indices", "")
            if not idx_str:
                continue
            for tok_s in idx_str.strip().split():
                if not tok_s:
                    continue
                try:
                    tok = int(tok_s)
                except ValueError:
                    # If for any reason an entry isn't an int, skip it
                    continue
                if SKIP_TOKENS and tok in SKIP_TOKENS:
                    continue
                counts[tok] += 1
                total_tokens += 1

    unique_codes = len(counts)
    ppl_e, ppl_2, H_e, H_2 = code_perplexity_from_counts(counts)

    print("=== Code Perplexity Report ===")
    print(f"CSV file:                {CSV_PATH}")
    print(f"Rows read:               {total_rows}")
    print(f"Total tokens (N):        {total_tokens}")
    print(f"Active/unique codes:     {unique_codes}")
    print(f"Entropy (nats):          {H_e:.6f}")
    print(f"Entropy (bits):          {H_2:.6f}")
    print(f"Code PPL (base e):       {ppl_e:.6f}")
    print(f"Code PPL (base 2):       {ppl_2:.6f}")
    if unique_codes > 0:
        print(f"Effective usage ratio:   {ppl_e/unique_codes:.6f}  (PPL_e / unique_codes)")

    # (Optional) show the top-10 most frequent codes
    topk = 10
    if unique_codes > 0:
        print(f"\nTop-{topk} codes by frequency:")
        for code, cnt in counts.most_common(topk):
            print(f"  code {code:>6}: {cnt}")

if __name__ == "__main__":
    main()
