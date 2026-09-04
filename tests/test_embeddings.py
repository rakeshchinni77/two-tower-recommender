"""Tests for Phase 3 item metadata, content embeddings, and baseline recommenders."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR,
    ITEMS_PATH,
    ITEM_EMBEDDINGS_PATH,
    EMBEDDING_MODEL_ID
)
from models.baselines import PopularityBaseline, ContentBaseline, PopularityRecommender, ContentRecommender


def test_items_csv_structure_and_integrity():
    assert ITEMS_PATH.exists(), f"items.csv missing at {ITEMS_PATH}"

    items_df = pd.read_csv(ITEMS_PATH)
    assert not items_df.empty, "items.csv is empty"

    required_cols = ["item_id", "title", "genres", "text"]
    for col in required_cols:
        assert col in items_df.columns, f"Missing required column '{col}' in items.csv"

    assert len(items_df) == 1682, f"Expected 1682 items, got {len(items_df)}"
    assert items_df["item_id"].nunique() == 1682, "item_id in items.csv is not unique"

    # Check text formatting
    sample_row = items_df.iloc[0]
    assert sample_row["title"] in sample_row["text"]


def test_item_embeddings_npy_matrix_quality():
    assert ITEM_EMBEDDINGS_PATH.exists(), f"item_embeddings.npy missing at {ITEM_EMBEDDINGS_PATH}"

    embeddings = np.load(ITEM_EMBEDDINGS_PATH)

    assert isinstance(embeddings, np.ndarray), "Loaded embeddings is not a NumPy ndarray"
    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
    assert embeddings.shape[0] == 1682, f"Expected 1682 rows, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == 384, f"Expected 384 dimensions, got {embeddings.shape[1]}"

    assert np.issubdtype(embeddings.dtype, np.floating), f"Expected float dtype, got {embeddings.dtype}"
    assert not np.isnan(embeddings).any(), "Embedding matrix contains NaN values"
    assert not np.isinf(embeddings).any(), "Embedding matrix contains Inf values"
    assert not np.all(embeddings == 0), "Embedding matrix is entirely zero"


def test_row_alignment_between_items_csv_and_embeddings():
    items_df = pd.read_csv(ITEMS_PATH)
    embeddings = np.load(ITEM_EMBEDDINGS_PATH)

    assert len(items_df) == embeddings.shape[0]

    # Verify first and last items map to valid non-zero vector embeddings
    first_emb = embeddings[0]
    last_emb = embeddings[-1]

    assert not np.all(first_emb == 0)
    assert not np.all(last_emb == 0)


def test_popularity_baseline_recommender():
    train_data = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [10, 20, 10, 30, 10],
        "rating": [5, 4, 5, 2, 4],
        "timestamp": [100, 101, 102, 103, 104]
    })

    pop_model = PopularityBaseline()
    pop_model.fit(train_data)

    recs = pop_model.recommend(top_k=2)
    # Item 10 has count=3, avg_rating=4.66 -> score ~ 14
    # Item 20 has count=1, avg_rating=4.0 -> score = 4
    # Item 30 has count=1, avg_rating=2.0 -> score = 2
    assert recs[0] == 10
    assert len(recs) == 2

    # Exclude item check
    recs_ex = pop_model.recommend(top_k=2, exclude_item_ids=[10])
    assert 10 not in recs_ex


def test_content_baseline_zero_history_handling():
    embeddings = np.random.randn(5, 384).astype(np.float32)
    item_ids = [1, 2, 3, 4, 5]

    pop_model = PopularityBaseline()
    pop_model.fit(pd.DataFrame({
        "user_id": [1, 1],
        "item_id": [1, 2],
        "rating": [5, 4],
        "timestamp": [100, 101]
    }))

    content_model = ContentBaseline(item_embeddings=embeddings, item_ids=item_ids, popularity_recommender=pop_model)

    # Without popularity fallback, history==[] returns empty list []
    content_no_pop = ContentBaseline(item_embeddings=embeddings, item_ids=item_ids)
    assert content_no_pop.recommend(history_item_ids=[], top_k=3) == []

    # With popularity fallback, zero history cascades safely
    assert content_model.recommend(history_item_ids=[], top_k=2) == [1, 2]


def test_content_baseline_recommendations():
    item_ids = [1, 2, 3, 4, 5]
    # Synthetic orthogonal embeddings for items
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    content_model = ContentBaseline(item_embeddings=embeddings, item_ids=item_ids)

    # User history includes item 1 (vector [1, 0, 0])
    recs = content_model.recommend(history_item_ids=[1], top_k=2)

    # Item 1 is excluded (in history). Item 2 has highest similarity (0.9 vs [1,0,0])
    assert recs[0] == 2
    assert 1 not in recs
