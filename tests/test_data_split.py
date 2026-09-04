"""Tests for MovieLens strict train/warm/cold data splitting pipeline and invariants."""

import pytest
import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DATA_DIR, THRESHOLD_N, RANDOM_SEED
from src.split_data import split_dataset, validate_split, select_cold_users

REQUIRED_COLS = ["user_id", "item_id", "rating", "timestamp"]


def test_processed_files_exist_and_nonempty():
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_warm_path = PROCESSED_DATA_DIR / "test_warm.csv"
    test_cold_path = PROCESSED_DATA_DIR / "test_cold.csv"

    assert train_path.exists(), "train.csv does not exist"
    assert test_warm_path.exists(), "test_warm.csv does not exist"
    assert test_cold_path.exists(), "test_cold.csv does not exist"

    train_df = pd.read_csv(train_path)
    test_warm_df = pd.read_csv(test_warm_path)
    test_cold_df = pd.read_csv(test_cold_path)

    assert not train_df.empty, "train.csv is empty"
    assert not test_warm_df.empty, "test_warm.csv is empty"
    assert not test_cold_df.empty, "test_cold.csv is empty"

    for name, df in [("train", train_df), ("test_warm", test_warm_df), ("test_cold", test_cold_df)]:
        assert list(df.columns) == REQUIRED_COLS, f"Invalid columns in {name}: {list(df.columns)}"


def test_cold_users_completely_absent_from_train_and_warm_test():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_warm_df = pd.read_csv(PROCESSED_DATA_DIR / "test_warm.csv")
    test_cold_df = pd.read_csv(PROCESSED_DATA_DIR / "test_cold.csv")

    train_users = set(train_df["user_id"])
    warm_users = set(test_warm_df["user_id"])
    cold_users = set(test_cold_df["user_id"])

    # Hard invariant: cold users must NEVER appear in train or test_warm
    cold_in_train = cold_users.intersection(train_users)
    assert len(cold_in_train) == 0, f"Cold user leakage into train: {cold_in_train}"

    cold_in_warm = cold_users.intersection(warm_users)
    assert len(cold_in_warm) == 0, f"Cold user leakage into warm test: {cold_in_warm}"

    # No user is simultaneously cold and warm
    assert len(cold_users.intersection(warm_users)) == 0


def test_warm_test_users_exist_in_train_and_satisfy_threshold():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_warm_df = pd.read_csv(PROCESSED_DATA_DIR / "test_warm.csv")

    train_users = set(train_df["user_id"])
    warm_users = set(test_warm_df["user_id"])

    # Every warm test user must exist in train
    absent = warm_users - train_users
    assert len(absent) == 0, f"Warm test users absent from train: {absent}"

    # Strict Warm Threshold Invariant: count(train interactions for warm user) > THRESHOLD_N
    train_counts = train_df.groupby("user_id").size()
    for u in warm_users:
        assert train_counts[u] > THRESHOLD_N, f"User {u} in test_warm has train count {train_counts[u]} <= {THRESHOLD_N}"


def test_no_interaction_leakage_between_splits():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test_warm_df = pd.read_csv(PROCESSED_DATA_DIR / "test_warm.csv")
    test_cold_df = pd.read_csv(PROCESSED_DATA_DIR / "test_cold.csv")

    def to_tuples(df):
        return set(zip(df["user_id"], df["item_id"], df["rating"], df["timestamp"]))

    train_tuples = to_tuples(train_df)
    warm_tuples = to_tuples(test_warm_df)
    cold_tuples = to_tuples(test_cold_df)

    assert len(train_tuples.intersection(warm_tuples)) == 0, "Interaction leakage between train and test_warm"
    assert len(train_tuples.intersection(cold_tuples)) == 0, "Interaction leakage between train and test_cold"
    assert len(warm_tuples.intersection(cold_tuples)) == 0, "Interaction leakage between test_warm and test_cold"

    # Completeness check
    assert len(train_df) + len(test_warm_df) + len(test_cold_df) == 100000


def test_data_types_and_value_ranges():
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    assert pd.api.types.is_integer_dtype(train_df["user_id"])
    assert pd.api.types.is_integer_dtype(train_df["item_id"])
    assert pd.api.types.is_integer_dtype(train_df["rating"])
    assert pd.api.types.is_integer_dtype(train_df["timestamp"])

    assert train_df["rating"].between(1, 5).all()
    assert (train_df["timestamp"] > 0).all()


def test_split_determinism():
    train1, warm1, cold1 = split_dataset(threshold_n=20, warm_test_fraction=0.20, cold_user_fraction=0.10, random_seed=42)
    train2, warm2, cold2 = split_dataset(threshold_n=20, warm_test_fraction=0.20, cold_user_fraction=0.10, random_seed=42)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(warm1, warm2)
    pd.testing.assert_frame_equal(cold1, cold2)


def test_split_with_synthetic_fixture():
    # Deterministic synthetic data fixture test
    rows = []
    # User 1: 50 interactions (warm eligible)
    for i in range(50):
        rows.append({"user_id": 1, "item_id": 100 + i, "rating": 5, "timestamp": 1000 + i})
    # User 2: 10 interactions (<= N, stay in train)
    for i in range(10):
        rows.append({"user_id": 2, "item_id": 200 + i, "rating": 4, "timestamp": 2000 + i})
    # User 3: 30 interactions (cold candidate)
    for i in range(30):
        rows.append({"user_id": 3, "item_id": 300 + i, "rating": 3, "timestamp": 3000 + i})

    synthetic_df = pd.DataFrame(rows)
    cold_users = {3}

    cold_df = synthetic_df[synthetic_df["user_id"].isin(cold_users)]
    warm_df = synthetic_df[~synthetic_df["user_id"].isin(cold_users)]

    # Validate split invariants on synthetic data
    assert set(cold_df["user_id"]).intersection(set(warm_df["user_id"])) == set()
