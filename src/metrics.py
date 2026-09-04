"""Evaluation metrics calculation module: Recall@K and NDCG@K."""

import math
from typing import Sequence, Any, Set


def _validate_k(k: int) -> None:
    """Validate top-K cutoff integer."""
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"K must be a positive integer, got {k}")


def recall_at_k(
    actual_items: Sequence[Any],
    predicted_items: Sequence[Any],
    k: int = 10
) -> float:
    """Calculate Recall@K:
    |predicted_top_K ∩ actual_test_items| / min(K, |actual_test_items|)

    Args:
        actual_items: Sequence of ground truth relevant item IDs.
        predicted_items: Sequence of model predicted item IDs ranked by score descending.
        k: Top-K cutoff integer (default 10).

    Returns:
        float score in range [0.0, 1.0].
    """
    _validate_k(k)

    if len(actual_items) == 0 or len(predicted_items) == 0:
        return 0.0

    actual_set: Set[Any] = set(actual_items)
    if len(actual_set) == 0:
        return 0.0

    # Evaluate only top-K predictions
    top_k_preds = list(predicted_items)[:k]

    # Count unique relevant items retrieved in top-K predictions
    hits = len(set(top_k_preds).intersection(actual_set))

    denominator = min(k, len(actual_set))
    if denominator == 0:
        return 0.0

    return float(hits / denominator)


def ndcg_at_k(
    actual_items: Sequence[Any],
    predicted_items: Sequence[Any],
    k: int = 10
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K (NDCG@K) for binary relevance.

    DCG@K = sum_{i=1..K} (relevance_i / log2(i + 1))
    IDCG@K = sum_{i=1..min(K, |actual_items|)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K

    Args:
        actual_items: Sequence of ground truth relevant item IDs.
        predicted_items: Sequence of model predicted item IDs ranked by score descending.
        k: Top-K cutoff integer (default 10).

    Returns:
        float score in range [0.0, 1.0].
    """
    _validate_k(k)

    if len(actual_items) == 0 or len(predicted_items) == 0:
        return 0.0

    actual_set: Set[Any] = set(actual_items)
    if len(actual_set) == 0:
        return 0.0

    # Calculate IDCG@K using min(K, len(actual_set))
    n_relevant = min(k, len(actual_set))
    if n_relevant == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    if idcg == 0.0:
        return 0.0

    # Calculate DCG@K up to top-K predictions, ensuring duplicate predictions count at most once
    dcg = 0.0
    seen_relevant: Set[Any] = set()
    top_k_preds = list(predicted_items)[:k]

    for rank_idx, item in enumerate(top_k_preds, start=1):
        if item in actual_set and item not in seen_relevant:
            dcg += 1.0 / math.log2(rank_idx + 1)
            seen_relevant.add(item)

    return float(dcg / idcg)


# Aliases
recall_k = recall_at_k
ndcg_k = ndcg_at_k
