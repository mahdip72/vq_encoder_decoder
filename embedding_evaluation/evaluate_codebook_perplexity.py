#!/usr/bin/env python3
# Read a CSV of VQ indices and compute:
# - Unigram entropy & perplexity
# - Zipf fit (unigram rank-frequency slope, R^2)
# - Bigram conditional entropy & perplexity; ΔH = H1 - H(Z_t|Z_{t-1})
# - Mutual information I(z_t; z_{t+Δ}) for selected lags
#
# CSV columns expected: pid, indices, protein_sequence
# 'indices' must be space-separated token IDs.
#
# Keep it simple: no argparse—configure constants below.

import csv
import math
from collections import Counter, defaultdict

# ========= Configure this =========
CSV_PATH = "path/to/vq_indices.csv"
SKIP_TOKENS = {-1}            # e.g., {-1} to ignore mask/padding
CODEBOOK_SIZE = 4096          # set to None to skip bound check
MI_LAGS = [1, 2, 4]           # lags at which to compute mutual information
TOPK = 10                     # how many unigrams/bigrams to print
# ==================================

def safe_int(tok_s):
    try:
        return int(tok_s)
    except Exception:
        return None

def valid_tok(tok):
    if tok is None:
        return False
    if SKIP_TOKENS and tok in SKIP_TOKENS:
        return False
    if CODEBOOK_SIZE is not None and (tok < 0 or tok >= CODEBOOK_SIZE):
        return False
    return True

def entropy_bits_from_counts(counter):
    N = sum(counter.values())
    if N == 0:
        return 0.0
    H = 0.0
    for c in counter.values():
        p = c / N
        H -= p * math.log2(p)
    return H

def perplexity_from_entropy_bits(H_bits):
    return 2.0 ** H_bits

def zipf_fit(counts):
    """
    Fit log(freq) = a + b*log(rank) over the positive-count unigrams.
    Returns (slope b, intercept a, R^2, n_points).
    """
    freqs = [c for c in counts.values() if c > 0]
    if not freqs:
        return 0.0, 0.0, 0.0, 0
    freqs.sort(reverse=True)
    # ranks start at 1
    xs = [math.log(r) for r in range(1, len(freqs) + 1)]
    ys = [math.log(f) for f in freqs]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, mean_y, 0.0, n
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    # R^2
    yhat = [intercept + slope * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 0.0 if ss_tot == 0 else (1.0 - ss_res / ss_tot)
    return slope, intercept, r2, n

def compute_mi_from_pairs(pair_counts):
    """
    pair_counts: dict[i] -> Counter({j: count_ij})
    Returns MI in bits computed from the pair sample (plugin estimate).
    """
    # total pairs
    T = 0
    row_sums = {}
    col_sums = Counter()
    for i, row in pair_counts.items():
        s = sum(row.values())
        row_sums[i] = s
        T += s
        for j, c in row.items():
            col_sums[j] += c
    if T == 0:
        return 0.0

    # MI = sum_{i,j} p_ij * log2( p_ij / (p_i * p_j) )
    I = 0.0
    for i, row in pair_counts.items():
        p_i = row_sums[i] / T
        for j, c in row.items():
            p_ij = c / T
            p_j = col_sums[j] / T
            # ensure numeric safety
            if p_ij > 0 and p_i > 0 and p_j > 0:
                I += p_ij * math.log2(p_ij / (p_i * p_j))
    return I

def main():
    # Unigram and bigram storage
    unigram = Counter()
    total_rows = 0
    total_tokens = 0

    # Bigram counts: prev -> Counter(next)
    bigram = defaultdict(Counter)
    total_bigrams = 0

    # Pair counts for MI at various lags: lag -> dict[i] -> Counter(j)
    lag_pair_counts = {d: defaultdict(Counter) for d in MI_LAGS}
    lag_totals = {d: 0 for d in MI_LAGS}

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            idx_str = row.get("indices", "")
            if not idx_str:
                continue
            # Parse & filter tokens for this sequence
            seq = []
            for tok_s in idx_str.strip().split():
                tok = safe_int(tok_s)
                if valid_tok(tok):
                    seq.append(tok)

            L = len(seq)
            if L == 0:
                continue

            # Unigrams
            unigram.update(seq)
            total_tokens += L

            # Bigram (lag=1) within the row
            for t in range(L - 1):
                i, j = seq[t], seq[t + 1]
                bigram[i][j] += 1
                total_bigrams += 1

            # MI at selected lags (within-row only)
            for d in MI_LAGS:
                if L <= d:
                    continue
                for t in range(L - d):
                    i, j = seq[t], seq[t + d]
                    lag_pair_counts[d][i][j] += 1
                    lag_totals[d] += 1

    # ==== Unigram stats ====
    H1_bits = entropy_bits_from_counts(unigram)
    PPL1 = perplexity_from_entropy_bits(H1_bits)
    active_codes = len(unigram)

    print("=== Token Statistics Report ===")
    print(f"CSV file:                    {CSV_PATH}")
    print(f"Rows read:                   {total_rows}")
    print(f"Total tokens (N):            {total_tokens}")
    print(f"Active/unique codes:         {active_codes}")
    print(f"Unigram entropy H1 (bits):   {H1_bits:.6f}")
    print(f"Code PPL (base 2):           {PPL1:.6f}")
    if active_codes > 0:
        print(f"Effective usage ratio:       {PPL1/active_codes:.6f}  (PPL1 / unique_codes)")

    # ==== Zipf fit on unigrams ====
    slope, intercept, r2, npts = zipf_fit(unigram)
    print("\n--- Unigram Zipf Fit ---")
    print(f"log(freq) ≈ a + b*log(rank)")
    print(f"b (slope):                   {slope:.4f}")
    print(f"a (intercept):               {intercept:.4f}")
    print(f"R^2:                         {r2:.4f}")
    print(f"n (ranks used):              {npts}")

    # ==== Bigram conditional entropy H(Z_t | Z_{t-1}) ====
    # Compute via expected conditional entropy weighted by p(i) estimated from bigram sample.
    if total_bigrams > 0:
        H2_given_1_bits = 0.0
        for i, row in bigram.items():
            row_total = sum(row.values())
            p_i = row_total / total_bigrams
            # conditional entropy for this context i
            H_cond_i = 0.0
            for c in row.values():
                p_j_given_i = c / row_total
                H_cond_i -= p_j_given_i * math.log2(p_j_given_i)
            H2_given_1_bits += p_i * H_cond_i

        PPL_cond = perplexity_from_entropy_bits(H2_given_1_bits)
        delta_bits = H1_bits - H2_given_1_bits  # = I(Z_{t-1}; Z_t)

        print("\n--- Bigram Conditional Statistics ---")
        print(f"Transitions counted:         {total_bigrams}")
        print(f"H(Z_t | Z_t-1) (bits):       {H2_given_1_bits:.6f}")
        print(f"Conditional PPL:             {PPL_cond:.6f}")
        print(f"ΔH = H1 - H2|1 (bits):       {delta_bits:.6f}  (Mutual information at lag 1)")
    else:
        print("\n--- Bigram Conditional Statistics ---")
        print("No bigrams counted (insufficient tokens).")

    # ==== Mutual Information at selected lags ====
    print("\n--- Mutual Information I(Z_t; Z_{t+Δ}) ---")
    if not MI_LAGS:
        print("No lags configured.")
    else:
        for d in MI_LAGS:
            T = lag_totals[d]
            if T == 0:
                print(f"Δ={d}:                       n/a (no pairs)")
                continue
            I_bits = compute_mi_from_pairs(lag_pair_counts[d])
            print(f"Δ={d}:                       {I_bits:.6f} bits  (pairs: {T})")

    # ==== Top-K reports ====
    print(f"\nTop-{TOPK} unigrams by frequency:")
    for code, cnt in unigram.most_common(TOPK):
        pct = (cnt / total_tokens * 100.0) if total_tokens > 0 else 0.0
        print(f"  code {code:>6}: {cnt}  ({pct:.4f}%)")

    print(f"\nTop-{TOPK} bigrams by frequency:")
    # Flatten bigrams for ranking
    bigram_flat = Counter()
    for i, row in bigram.items():
        for j, c in row.items():
            bigram_flat[(i, j)] = c
    for (i, j), c in bigram_flat.most_common(TOPK):
        pct = (c / total_bigrams * 100.0) if total_bigrams > 0 else 0.0
        print(f"  ({i:>4} -> {j:>4}): {c}  ({pct:.4f}%)")

if __name__ == "__main__":
    main()
