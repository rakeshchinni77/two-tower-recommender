"""Unit tests for Phase 1 data loader and dataset validation."""

import pytest
import pandas as pd
from src.data_loader import (
    load_interactions,
    load_items,
    load_users,
    load_dataset,
    validate_dataset
)
from src.config import RAW_DATA_DIR


def test_load_interactions_movielens100k():
    df = load_interactions(RAW_DATA_DIR)
    assert not df.empty
    assert len(df) == 100000
    assert list(df.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert df["user_id"].nunique() == 943
    assert df["item_id"].nunique() == 1682
    assert df["rating"].min() == 1
    assert df["rating"].max() == 5
    assert df["timestamp"].min() > 0


def test_load_items_movielens100k():
    items = load_items(RAW_DATA_DIR)
    assert not items.empty
    assert len(items) == 1682
    assert "item_id" in items.columns
    assert "title" in items.columns
    assert "genres" in items.columns
    # Check sample movie titles readable without character corruption
    sample_title = items.loc[items["item_id"] == 1, "title"].values[0]
    assert "Toy Story" in sample_title


def test_load_users_movielens100k():
    users = load_users(RAW_DATA_DIR)
    assert not users.empty
    assert len(users) == 943
    assert "user_id" in users.columns


def test_validate_dataset_integrity():
    interactions, items = load_dataset(RAW_DATA_DIR)
    stats = validate_dataset(interactions, items)
    assert stats["interaction_count"] == 100000
    assert stats["unique_users"] == 943
    assert stats["unique_items_in_interactions"] == 1682
    assert stats["join_validation_passed"] is True
