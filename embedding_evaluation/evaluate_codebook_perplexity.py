# Compute:
# - Unigram entropy & perplexity
# - Unigram Zipf fit
# - Bigram conditional entropy & perplexity; ΔH1 = H1 - H2|1
# - Trigram conditional entropy & perplexity; ΔH2 = H1 - H3|21; extra gain = H2|1 - H3|21
# - Mutual information I(Z_t; Z_{t+Δ}) at selected lags
#
# CSV columns expected: pid, indices, protein_sequence
# 'indices' must be space-separated token IDs.

import csv
import math
from collections import Counter, defaultdict
import os
import sys
from datetime import datetime

# ========= Configure this =========
CSV_PATH = "../inference_encode_results/validation_set_2048_h5/vqvae/vq_indices.csv"
SKIP_TOKENS = {-1}            # e.g., {-1} to ignore mask/padding
CODEBOOK_SIZE = 4096          # set to None to skip bound check
MI_LAGS = [1, 2, 4]           # lags for MI computation
TOPK = 10                     # how many unigrams/bigrams/trigrams to print
# ==================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

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
    freqs = [c for c in counts.values() if c > 0]
    if not freqs:
        return 0.0, 0.0, 0.0, 0
    freqs.sort(reverse=True)
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
    yhat = [intercept + slope * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 0.0 if ss_tot == 0 else (1.0 - ss_res / ss_tot)
    return slope, intercept, r2, n

def compute_mi_from_pairs(pair_counts):
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
    I = 0.0
    for i, row in pair_counts.items():
        p_i = row_sums[i] / T
        for j, c in row.items():
            p_ij = c / T
            p_j = col_sums[j] / T
            if p_ij > 0 and p_i > 0 and p_j > 0:
                I += p_ij * math.log2(p_ij / (p_i * p_j))
    return I

def main():
    # Setup log file beside the input CSV; tee stdout to both console and file
    log_dir = os.path.dirname(os.path.abspath(CSV_PATH))
    log_path = os.path.join(log_dir, "codebook_analysis.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, log_file)

    # Unigram
    unigram = Counter()
    total_rows = 0
    total_tokens = 0

    # Bigram: prev -> Counter(next)
    bigram = defaultdict(Counter)
    total_bigrams = 0

    # Trigram: (prev2, prev1) -> Counter(next)
    trigram = defaultdict(Counter)
    total_trigrams = 0

    # MI at selected lags: lag -> dict[i] -> Counter(j)
    lag_pair_counts = {d: defaultdict(Counter) for d in MI_LAGS}
    lag_totals = {d: 0 for d in MI_LAGS}

    # Read CSV
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            idx_str = row.get("indices", "")
            if not idx_str:
                continue
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

            # Bigrams
            for t in range(L - 1):
                i, j = seq[t], seq[t + 1]
                bigram[i][j] += 1
                total_bigrams += 1

            # Trigrams
            for t in range(L - 2):
                i, j, k = seq[t], seq[t + 1], seq[t + 2]
                trigram[(i, j)][k] += 1
                total_trigrams += 1

            # MI lags
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
    print("log(freq) ≈ a + b*log(rank)")
    print(f"b (slope):                   {slope:.4f}")
    print(f"a (intercept):               {intercept:.4f}")
    print(f"R^2:                         {r2:.4f}")
    print(f"n (ranks used):              {npts}")

    # ==== Bigram conditional entropy H(Z_t | Z_{t-1}) ====
    if total_bigrams > 0:
        H2_given_1_bits = 0.0
        for i, row in bigram.items():
            row_total = sum(row.values())
            p_i = row_total / total_bigrams
            H_cond_i = 0.0
            for c in row.values():
                p_j_given_i = c / row_total
                H_cond_i -= p_j_given_i * math.log2(p_j_given_i)
            H2_given_1_bits += p_i * H_cond_i
        PPL2cond = perplexity_from_entropy_bits(H2_given_1_bits)
        delta1_bits = H1_bits - H2_given_1_bits  # I(Z_{t-1}; Z_t)

        print("\n--- Bigram Conditional Statistics ---")
        print(f"Transitions counted:         {total_bigrams}")
        print(f"H(Z_t | Z_{{t-1}}) (bits):     {H2_given_1_bits:.6f}")
        print(f"Conditional PPL:             {PPL2cond:.6f}")
        print(f"ΔH1 = H1 - H(Z_t|Z_{{t-1}}):   {delta1_bits:.6f}  (I(Z_{{t-1}}; Z_t))")
    else:
        H2_given_1_bits = float("nan")
        PPL2cond = float("nan")
        delta1_bits = float("nan")
        print("\n--- Bigram Conditional Statistics ---")
        print("No bigrams counted.")

    # ==== Trigram conditional entropy H(Z_t | Z_{t-2}, Z_{t-1}) ====
    if total_trigrams > 0:
        H3_given_21_bits = 0.0
        for ctx, row in trigram.items():
            row_total = sum(row.values())
            p_ctx = row_total / total_trigrams            # p(i,j)
            H_cond_ctx = 0.0
            for c in row.values():
                p_k_given_ctx = c / row_total             # p(k | i,j)
                H_cond_ctx -= p_k_given_ctx * math.log2(p_k_given_ctx)
            H3_given_21_bits += p_ctx * H_cond_ctx

        PPL3cond = perplexity_from_entropy_bits(H3_given_21_bits)
        delta2_bits = H1_bits - H3_given_21_bits          # I((Z_{t-2},Z_{t-1}); Z_t)
        extra_gain_bits = H2_given_1_bits - H3_given_21_bits  # additional info from Z_{t-2}

        print("\n--- Trigram Conditional Statistics ---")
        print(f"H(Z_t | Z_{{t-2}}, Z_{{t-1}}) (bits):  {H3_given_21_bits:.6f}")
        print(f"Conditional PPL:                   {PPL3cond:.6f}")
        print(f"ΔH2 = H1 - H(Z_t|Z_{{t-2}},Z_{{t-1}}): {delta2_bits:.6f}  (I((Z_{{t-2}},Z_{{t-1}}); Z_t))")
        print(f"Extra gain over bigram:            {extra_gain_bits:.6f}  (= H(Z_t|Z_{{t-1}}) - H(Z_t|Z_{{t-2}},Z_{{t-1}}))")
    else:
        print("\n--- Trigram Conditional Statistics ---")
        print("No trigrams counted.")

    print(f"Trigram transitions counted:  {total_trigrams}")
    print(f"Unique trigram contexts:      {len(trigram)}")

    # ==== Mutual Information at selected lags ====
    print("\n--- Mutual Information I(Z_t; Z_{t+Δ}) ---")
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
    bigram_flat = Counter()
    for i, row in bigram.items():
        for j, c in row.items():
            bigram_flat[(i, j)] = c
    for (i, j), c in bigram_flat.most_common(TOPK):
        pct = (c / total_bigrams * 100.0) if total_bigrams > 0 else 0.0
        print(f"  ({i:>4} -> {j:>4}): {c}  ({pct:.4f}%)")

    print(f"\nTop-{TOPK} trigrams by frequency:")
    trigram_flat = Counter()
    for (i, j), row in trigram.items():
        for k, c in row.items():
            trigram_flat[(i, j, k)] = c
    for (i, j, k), c in trigram_flat.most_common(TOPK):
        pct = (c / total_trigrams * 100.0) if total_trigrams > 0 else 0.0
        print(f"  ({i:>4} -> {j:>4} -> {k:>4}): {c}  ({pct:.4f}%)")

    print(f"\nSaved log to: {log_path}")
    log_file.flush()
    sys.stdout = old_stdout
    log_file.close()

if __name__ == "__main__":
    main()
