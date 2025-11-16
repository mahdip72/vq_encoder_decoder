"""
Utilities for computing codebook usage statistics during validation.

This module contains the :class:`CodebookUsageStats` metric, which mirrors the
standalone ``embedding_evaluation/evaluate_codebook_perplexity.py`` script but
is implemented as a TorchMetrics object so it integrates seamlessly with the
training loop. It maintains host-side counters for:

* Unigram frequencies and Zipf regression
* Bigram / trigram conditional entropies and perplexities
* ΔH improvements (I(Z_{t-1}; Z_t), I((Z_{t-2}, Z_{t-1}); Z_t), and the extra
  gain from Z_{t-2})
* Mutual information at fixed lags (currently 1, 2, and 4)
* Effective usage ratio (unigram perplexity divided by the number of active
  codes)

During validation we gather the token indices across ranks, update this metric
once per batch on rank 0, and at epoch end log all scalars to TensorBoard under
``codebook_usage_statistics/*``. The class purposely avoids torch tensors for
its internal counters so the arithmetic matches the reference implementation and
does not require extra synchronization beyond the gathered indices.
"""

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from torchmetrics import Metric


def _entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _perplexity_from_entropy(entropy_bits: float) -> float:
    return float(2.0 ** entropy_bits)


def _zipf_fit(counts: Counter) -> Tuple[float, float, float, int]:
    """Linear fit log(freq) ~ intercept + slope * log(rank)."""
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


def _mutual_information_bits(pair_counts: Dict[int, Counter]) -> float:
    total = 0
    row_sums: Dict[int, int] = {}
    col_sums: Counter = Counter()
    for i, row in pair_counts.items():
        row_total = sum(row.values())
        row_sums[i] = row_total
        total += row_total
        for j, count in row.items():
            col_sums[j] += count

    if total == 0:
        return 0.0

    mi = 0.0
    for i, row in pair_counts.items():
        p_i = row_sums[i] / total if total > 0 else 0.0
        for j, count in row.items():
            if count <= 0:
                continue
            p_ij = count / total
            p_j = col_sums[j] / total
            if p_ij > 0 and p_i > 0 and p_j > 0:
                mi += p_ij * math.log2(p_ij / (p_i * p_j))
    return mi


class CodebookUsageStats(Metric):
    """TorchMetrics helper that captures detailed VQ code statistics in validation.

    The metric accumulates counts for unigrams, bigrams, trigrams, and fixed-lag
    pairs while ignoring padding tokens (default ``-1``) and any invalid codes
    outside ``[0, codebook_size)``. All counters live on the CPU because we call
    ``accelerator.gather_for_metrics`` beforehand; as a result, the implementation
    is deterministic and matches the offline evaluator.

    Args:
        codebook_size: Number of entries in the codebook. Used to filter out
            indices outside the valid range; pass ``None`` to skip the bound
            check entirely.
        mi_lags: Iterable of positive integers specifying the Δ offsets used for
            mutual-information computation (e.g., ``(1, 2, 4)``).
        skip_tokens: Token IDs to drop before counting (default is the padding
            sentinel ``-1``).
        dist_sync_on_step: TorchMetrics flag; we keep the default ``False`` since
            this metric runs entirely on the main process after gathering.
    """

    full_state_update = False

    def __init__(
        self,
        codebook_size: int,
        mi_lags: Sequence[int] = (1, 2, 4),
        skip_tokens: Iterable[int] = (-1,),
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.codebook_size = codebook_size
        self.mi_lags: Tuple[int, ...] = tuple(sorted(set(int(d) for d in mi_lags if d > 0)))
        self.skip_tokens = set(skip_tokens or [])
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.unigram: Counter = Counter()
        self.bigram: Dict[int, Counter] = defaultdict(Counter)
        self.trigram: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self.total_tokens = 0
        self.total_bigrams = 0
        self.total_trigrams = 0
        self.mi_pair_counts: Dict[int, Dict[int, Counter]] = {
            lag: defaultdict(Counter) for lag in self.mi_lags
        }
        self.mi_totals: Dict[int, int] = {lag: 0 for lag in self.mi_lags}

    def update(self, indices) -> None:
        """Accumulate counts for a gathered batch of indices.

        Args:
            indices: A tensor-like object or nested list ``(B, L)`` representing
                codebook IDs. The caller is responsible for running
                ``accelerator.gather_for_metrics`` so that every replica
                contributes to the statistics before we update on rank 0.
        """
        if indices is None:
            return
        if hasattr(indices, "detach"):
            sequences = indices.detach().cpu().tolist()
        else:
            sequences = indices
        if not isinstance(sequences, list):
            return
        for seq in sequences:
            filtered = self._filter_sequence(seq)
            length = len(filtered)
            if length == 0:
                continue
            self.unigram.update(filtered)
            self.total_tokens += length

            if length > 1:
                for t in range(length - 1):
                    i, j = filtered[t], filtered[t + 1]
                    self.bigram[i][j] += 1
                    self.total_bigrams += 1

            if length > 2:
                for t in range(length - 2):
                    ctx = (filtered[t], filtered[t + 1])
                    nxt = filtered[t + 2]
                    self.trigram[ctx][nxt] += 1
                    self.total_trigrams += 1

            for lag in self.mi_lags:
                if length <= lag:
                    continue
                pair_counts = self.mi_pair_counts[lag]
                total_pairs = 0
                for t in range(length - lag):
                    i = filtered[t]
                    j = filtered[t + lag]
                    pair_counts[i][j] += 1
                    total_pairs += 1
                self.mi_totals[lag] += total_pairs

    def compute(self) -> Dict[str, float]:
        """Return the scalar statistics derived from accumulated counts."""
        stats: Dict[str, float] = {}

        entropy_unigram = _entropy_bits(self.unigram)
        perplexity_unigram = _perplexity_from_entropy(entropy_unigram)
        stats["entropy_unigram_bits"] = entropy_unigram
        stats["perplexity_unigram"] = perplexity_unigram

        active_codes = len(self.unigram)
        stats["active_codes"] = float(active_codes)
        stats["effective_usage_ratio"] = (
            perplexity_unigram / active_codes if active_codes > 0 else float("nan")
        )

        entropy_bigram = self._conditional_entropy(self.bigram, self.total_bigrams)
        stats["entropy_bigram_cond_bits"] = entropy_bigram
        stats["perplexity_bigram"] = (
            _perplexity_from_entropy(entropy_bigram) if math.isfinite(entropy_bigram) else float("nan")
        )

        entropy_trigram = self._trigram_entropy()
        stats["entropy_trigram_cond_bits"] = entropy_trigram
        stats["perplexity_trigram"] = (
            _perplexity_from_entropy(entropy_trigram) if math.isfinite(entropy_trigram) else float("nan")
        )

        stats["delta_entropy_h1_h2"] = (
            entropy_unigram - entropy_bigram if math.isfinite(entropy_bigram) else float("nan")
        )
        stats["delta_entropy_h1_h3"] = (
            entropy_unigram - entropy_trigram if math.isfinite(entropy_trigram) else float("nan")
        )
        stats["delta_entropy_conditional"] = (
            entropy_bigram - entropy_trigram
            if math.isfinite(entropy_bigram) and math.isfinite(entropy_trigram)
            else float("nan")
        )

        slope, _, r2, _ = _zipf_fit(self.unigram)
        stats["zipf_slope"] = slope
        stats["zipf_r2"] = r2

        for lag in self.mi_lags:
            mi_value = (
                _mutual_information_bits(self.mi_pair_counts[lag]) if self.mi_totals[lag] > 0 else float("nan")
            )
            stats[f"mutual_info_lag{lag}"] = mi_value

        return stats

    def _filter_sequence(self, seq: Sequence[int]) -> List[int]:
        """Remove invalid tokens from a raw sequence prior to counting."""
        if not isinstance(seq, Sequence):
            return []
        filtered = []
        for tok in seq:
            if tok is None:
                continue
            val = int(tok)
            if val in self.skip_tokens:
                continue
            if self.codebook_size is not None and (val < 0 or val >= self.codebook_size):
                continue
            filtered.append(val)
        return filtered

    def _conditional_entropy(self, counts: Dict[int, Counter], total: int) -> float:
        """Compute conditional entropy for bigrams given total transitions."""
        if total == 0:
            return float("nan")
        entropy = 0.0
        for _, row in counts.items():
            row_total = sum(row.values())
            if row_total == 0:
                continue
            p_ctx = row_total / total
            ctx_entropy = 0.0
            for c in row.values():
                if c <= 0:
                    continue
                p = c / row_total
                ctx_entropy -= p * math.log2(p)
            entropy += p_ctx * ctx_entropy
        return entropy

    def _trigram_entropy(self) -> float:
        """Compute trigram conditional entropy H(Z_t | Z_{t-2}, Z_{t-1})."""
        if self.total_trigrams == 0:
            return float("nan")
        entropy = 0.0
        for (_, _), row in self.trigram.items():
            row_total = sum(row.values())
            if row_total == 0:
                continue
            p_ctx = row_total / self.total_trigrams
            ctx_entropy = 0.0
            for c in row.values():
                if c <= 0:
                    continue
                p = c / row_total
                ctx_entropy -= p * math.log2(p)
            entropy += p_ctx * ctx_entropy
        return entropy
