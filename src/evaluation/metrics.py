"""Offline evaluation metrics for retrieval and ranking."""
from __future__ import annotations

import numpy as np


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Recall@K: fraction of relevant items in top-K retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    """NDCG@K for binary relevance."""
    top_k = ranked_items[:k]
    gains = np.array([1.0 if item in relevant else 0.0 for item in top_k])
    discounts = np.log2(np.arange(len(gains)) + 2)
    dcg = np.sum(gains / discounts)

    ideal_gains = np.sort(gains)[::-1]
    idcg = np.sum(ideal_gains / np.log2(np.arange(len(ideal_gains)) + 2))

    return float(dcg / idcg) if idcg > 0 else 0.0


def coverage(all_recommended: set[int], all_items: set[int]) -> float:
    """Catalog coverage: fraction of items ever recommended."""
    if not all_items:
        return 0.0
    return len(all_recommended) / len(all_items)


def diversity_entropy(genre_counts: dict[str, int]) -> float:
    """Genre diversity via entropy of genre distribution in recommendations."""
    total = sum(genre_counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in genre_counts.values()])
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))
