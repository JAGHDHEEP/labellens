"""Ranking and classification metrics. Pure functions, unit-tested."""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    return 1.0 if any(r in gold for r in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    for i, r in enumerate(ranked[:k], start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = (len(xs) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score of a random positive > score of a random negative); ties count half."""
    if not pos or not neg:
        return 0.0
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_ci(values: Sequence[float], iters: int = 1000, seed: int = 7) -> tuple[float, float]:
    """95 % bootstrap confidence interval of the mean - eval sets here are a few
    hundred items, so a 2-point difference is often inside the noise."""
    import random

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters) - 1])
