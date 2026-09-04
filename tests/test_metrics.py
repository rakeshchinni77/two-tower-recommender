"""Comprehensive unit tests for evaluation metrics (Recall@K, NDCG@K)."""

import math
import pytest
import numpy as np

from src.metrics import recall_at_k, ndcg_at_k


def test_recall_manual_example():
    actual = [1, 2, 3]
    predicted = [3, 9, 8, 1]
    k = 4

    # Top 4 predicted = [3, 9, 8, 1], hits = {1, 3} (2 hits)
    # denominator = min(4, len(actual)) = 3
    # Expected Recall@4 = 2 / 3
    recall = recall_at_k(actual, predicted, k=k)
    assert recall == pytest.approx(2.0 / 3.0)


def test_ndcg_manual_example():
    actual = [1, 2, 3]
    predicted = [3, 9, 8, 1]
    k = 4

    # Rank 1: item 3 -> rel = 1 -> 1 / log2(2) = 1.0
    # Rank 2: item 9 -> rel = 0
    # Rank 3: item 8 -> rel = 0
    # Rank 4: item 1 -> rel = 1 -> 1 / log2(5)
    expected_dcg = (1.0 / math.log2(2)) + (1.0 / math.log2(5))

    # IDCG for 3 relevant items at K=4:
    # Rank 1 -> 1/log2(2), Rank 2 -> 1/log2(3), Rank 3 -> 1/log2(4)
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3)) + (1.0 / math.log2(4))

    expected_ndcg = expected_dcg / expected_idcg

    ndcg = ndcg_at_k(actual, predicted, k=k)
    assert ndcg == pytest.approx(expected_ndcg)


def test_recall_perfect_prediction():
    actual = [1, 2, 3]
    predicted = [1, 2, 3]
    assert recall_at_k(actual, predicted, k=10) == 1.0


def test_ndcg_perfect_prediction():
    actual = [1, 2, 3]
    predicted = [1, 2, 3]
    assert ndcg_at_k(actual, predicted, k=10) == 1.0


def test_recall_no_overlap():
    actual = [1, 2, 3]
    predicted = [8, 9, 10]
    assert recall_at_k(actual, predicted, k=10) == 0.0


def test_ndcg_no_overlap():
    actual = [1, 2, 3]
    predicted = [8, 9, 10]
    assert ndcg_at_k(actual, predicted, k=10) == 0.0


def test_empty_actual():
    actual = []
    predicted = [1, 2, 3]
    assert recall_at_k(actual, predicted, k=10) == 0.0
    assert ndcg_at_k(actual, predicted, k=10) == 0.0


def test_empty_predictions():
    actual = [1, 2, 3]
    predicted = []
    assert recall_at_k(actual, predicted, k=10) == 0.0
    assert ndcg_at_k(actual, predicted, k=10) == 0.0


def test_k_smaller_than_relevant_items():
    actual = [1, 2, 3, 4, 5]
    predicted = [1, 2, 3, 4, 5]
    k = 2

    # min(2, 5) = 2. Hits in top 2 = 2. Recall@2 = 2 / 2 = 1.0
    assert recall_at_k(actual, predicted, k=k) == 1.0
    assert ndcg_at_k(actual, predicted, k=k) == 1.0


def test_predictions_outside_top_k():
    actual = [1]
    predicted = [9, 8, 7, 6, 5, 4, 3, 2, 1, 10]
    k = 5

    # Item 1 is at index 8 (rank 9), outside top 5
    assert recall_at_k(actual, predicted, k=k) == 0.0
    assert ndcg_at_k(actual, predicted, k=k) == 0.0


def test_duplicate_predictions():
    actual = [1, 2, 3]
    predicted = [1, 1, 1, 8, 9]
    k = 5

    # Top 5 preds = [1, 1, 1, 8, 9]. Unique hit = {1} (1 hit).
    # Denominator = min(5, 3) = 3 -> Recall@5 = 1 / 3
    assert recall_at_k(actual, predicted, k=k) == pytest.approx(1.0 / 3.0)

    # NDCG@5: Rank 1 gives 1/log2(2). Ranks 2 and 3 duplicate item 1 -> 0 gain.
    # IDCG@5 for 3 items = 1/log2(2) + 1/log2(3) + 1/log2(4)
    dcg = 1.0 / math.log2(2)
    idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    expected_ndcg = dcg / idcg

    assert ndcg_at_k(actual, predicted, k=k) == pytest.approx(expected_ndcg)


def test_invalid_k():
    actual = [1, 2]
    predicted = [1, 2]

    with pytest.raises(ValueError):
        recall_at_k(actual, predicted, k=0)

    with pytest.raises(ValueError):
        ndcg_at_k(actual, predicted, k=-5)


def test_numpy_array_and_tuple_inputs():
    actual = np.array([10, 20, 30])
    predicted = (30, 90, 80, 10)

    assert recall_at_k(actual, predicted, k=4) == pytest.approx(2.0 / 3.0)
    assert ndcg_at_k(actual, predicted, k=4) > 0.0
