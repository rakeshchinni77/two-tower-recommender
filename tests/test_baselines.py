"""Comprehensive tests for Phase 4 baseline recommenders (Popularity & Content-Similarity)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, ITEMS_PATH, ITEM_EMBEDDINGS_PATH
from models.baselines import PopularityBaseline, ContentBaseline, PopularityRecommender, ContentRecommender


def test_popularity_baseline_on_movielens_train_csv():
    train_path = PROCESSED_DATA_DIR / "train.csv"
    assert train_path.exists(), f"train.csv missing at {train_path}"

    train_df = pd.read_csv(train_path)
    pop_model = PopularityBaseline()
    pop_model.fit(train_df)

    recs = pop_model.recommend(top_k=10)

    assert len(recs) == 10
    assert all(isinstance(x, int) for x in recs)
    assert len(set(recs)) == 10, "Popularity recommendations contain duplicates"

    # Verify score formula on train dataset top item
    top_item = recs[0]
    item_df = train_df[train_df["item_id"] == top_item]
    expected_count = len(item_df)
    expected_avg = item_df["rating"].mean()
    expected_score = expected_count * expected_avg

    assert abs(pop_model.item_scores[top_item] - expected_score) < 1e-5


def test_popularity_formula_synthetic_fixture():
    # Synthetic fixture for manual calculation check
    synthetic_train = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 2, 1, 2, 3, 4],
        "item_id": [10, 10, 10, 20, 20, 30, 30, 30, 30],
        "rating": [4, 5, 3, 5, 5, 1, 1, 1, 1],
        "timestamp": list(range(100, 109))
    })

    # Item 10: count=3, ratings=[4,5,3], avg=4.0 -> pop_score = 3 * 4.0 = 12.0
    # Item 20: count=2, ratings=[5,5], avg=5.0 -> pop_score = 2 * 5.0 = 10.0
    # Item 30: count=4, ratings=[1,1,1,1], avg=1.0 -> pop_score = 4 * 1.0 = 4.0

    pop_model = PopularityRecommender().fit(synthetic_train)

    assert pop_model.item_scores[10] == 12.0
    assert pop_model.item_scores[20] == 10.0
    assert pop_model.item_scores[30] == 4.0

    recs = pop_model.recommend(top_k=3)
    assert recs == [10, 20, 30]


def test_content_baseline_on_movielens_processed_data():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    pop_model = PopularityRecommender().fit(train_df)
    content_model = ContentBaseline.load_from_processed(PROCESSED_DATA_DIR, popularity_recommender=pop_model)

    sample_history = [1, 50, 100]  # Toy Story, Star Wars, etc.
    recs = content_model.recommend(history_item_ids=sample_history, top_k=10)

    assert len(recs) == 10
    assert all(isinstance(x, int) for x in recs)
    assert len(set(recs)) == 10
    # History items must be excluded
    for item in sample_history:
        assert item not in recs


def test_content_profile_arithmetic_mean_synthetic():
    item_ids = [101, 102, 103]
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # Item 101
        [0.0, 1.0, 0.0],  # Item 102
        [0.5, 0.5, 0.0],  # Item 103 (closer to mean of 101 and 102)
    ], dtype=np.float32)

    content_model = ContentRecommender(item_embeddings=embeddings, item_ids=item_ids)

    # History: [101, 102]. Profile mean = [0.5, 0.5, 0.0]
    # Similarity with 103 ([0.5, 0.5, 0.0]) will be 1.0
    recs = content_model.recommend(history_item_ids=[101, 102], top_k=1)

    assert recs == [103]


def test_exclusion_logic():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    pop_model = PopularityRecommender().fit(train_df)

    top_popular = pop_model.recommend(top_k=5)
    exclude_target = top_popular[0]

    recs_ex = pop_model.recommend(top_k=5, exclude_items=[exclude_target])
    assert exclude_target not in recs_ex

    content_model = ContentBaseline.load_from_processed(PROCESSED_DATA_DIR, popularity_recommender=pop_model)
    recs_content_ex = content_model.recommend(history_item_ids=[10], top_k=5, exclude_items=[50, 100])

    assert 10 not in recs_content_ex
    assert 50 not in recs_content_ex
    assert 100 not in recs_content_ex


def test_zero_history_routing_to_popularity():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    pop_model = PopularityRecommender().fit(train_df)
    content_model = ContentBaseline.load_from_processed(PROCESSED_DATA_DIR, popularity_recommender=pop_model)

    pop_top_10 = pop_model.recommend(top_k=10)
    zero_hist_recs = content_model.recommend(history_item_ids=[], top_k=10)

    assert zero_hist_recs == pop_top_10


def test_unknown_item_ids_and_duplicates_handling():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    pop_model = PopularityRecommender().fit(train_df)
    content_model = ContentBaseline.load_from_processed(PROCESSED_DATA_DIR, popularity_recommender=pop_model)

    # 999999 is invalid/unknown item ID. Duplicate 1 is present.
    recs = content_model.recommend(history_item_ids=[999999, 1, 1, 999998], top_k=5)

    assert len(recs) == 5
    assert all(isinstance(x, int) for x in recs)
    assert 1 not in recs


def test_zero_norm_embedding_safety():
    item_ids = [1, 2, 3]
    embeddings = np.array([
        [0.0, 0.0, 0.0],  # Zero norm vector
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)

    content_model = ContentRecommender(item_embeddings=embeddings, item_ids=item_ids)
    recs = content_model.recommend(history_item_ids=[2], top_k=2)

    assert len(recs) == 2
    assert 2 not in recs
    assert set(recs) == {1, 3}
