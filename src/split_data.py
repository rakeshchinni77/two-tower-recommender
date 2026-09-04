"""Strict MovieLens train/warm/cold data splitting module."""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Set, Any, Union
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    THRESHOLD_N,
    WARM_TEST_FRACTION,
    COLD_USER_FRACTION,
    RANDOM_SEED
)
from src.data_loader import load_interactions

REQUIRED_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def select_cold_users(
    unique_users: list,
    cold_user_fraction: float = COLD_USER_FRACTION,
    random_seed: int = RANDOM_SEED
) -> Set[int]:
    """Deterministically select a random subset of users to be held out as cold users."""
    sorted_users = sorted(unique_users)
    num_cold = int(len(sorted_users) * cold_user_fraction)
    rng = np.random.RandomState(random_seed)
    cold_users = set(rng.choice(sorted_users, size=num_cold, replace=False).tolist())
    return cold_users


def split_warm_user_interactions(
    user_df: pd.DataFrame,
    warm_test_fraction: float = WARM_TEST_FRACTION,
    threshold_n: int = THRESHOLD_N
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions for a single non-cold user into train and test_warm partitions.

    Ensures that if test interactions are held out, the user retains > threshold_n interactions in train.
    """
    sorted_df = user_df.sort_values(by=["timestamp", "item_id"]).reset_index(drop=True)
    total_count = len(sorted_df)

    num_test = int(np.round(total_count * warm_test_fraction))
    if num_test < 1:
        num_test = 1
    num_train = total_count - num_test

    # Strict Warm-State Invariant: count(train interactions) > THRESHOLD_N
    if num_train > threshold_n and num_test >= 1:
        train_part = sorted_df.iloc[:num_train]
        test_part = sorted_df.iloc[num_train:]
        return train_part, test_part
    else:
        # Cannot satisfy > threshold_n in train if test interactions are held out; place all in train
        empty_test = pd.DataFrame(columns=REQUIRED_COLUMNS).astype({
            "user_id": int, "item_id": int, "rating": int, "timestamp": int
        })
        return sorted_df, empty_test


def validate_split(
    train_df: pd.DataFrame,
    test_warm_df: pd.DataFrame,
    test_cold_df: pd.DataFrame,
    threshold_n: int = THRESHOLD_N,
    total_raw_count: int = 100000
) -> Dict[str, Any]:
    """Validate all hard data-splitting invariants and check for data leakage."""
    train_users = set(train_df["user_id"].unique()) if not train_df.empty else set()
    warm_test_users = set(test_warm_df["user_id"].unique()) if not test_warm_df.empty else set()
    cold_users = set(test_cold_df["user_id"].unique()) if not test_cold_df.empty else set()

    # 1. Cold user leakage checks
    cold_in_train = cold_users.intersection(train_users)
    if cold_in_train:
        raise ValueError(f"HARD INVARIANT VIOLATION: Cold users present in train.csv: {cold_in_train}")

    cold_in_warm_test = cold_users.intersection(warm_test_users)
    if cold_in_warm_test:
        raise ValueError(f"HARD INVARIANT VIOLATION: Cold users present in test_warm.csv: {cold_in_warm_test}")

    # 2. Warm test user presence in train check
    warm_absent_from_train = warm_test_users - train_users
    if warm_absent_from_train:
        raise ValueError(f"HARD INVARIANT VIOLATION: Warm test users absent from train.csv: {warm_absent_from_train}")

    # 3. Strict Warm Threshold Invariant check: count(train interactions) > THRESHOLD_N
    invalid_warm_users = []
    if not train_df.empty and warm_test_users:
        train_counts = train_df.groupby("user_id").size()
        for u in warm_test_users:
            if train_counts.get(u, 0) <= threshold_n:
                invalid_warm_users.append((u, train_counts.get(u, 0)))

    if invalid_warm_users:
        raise ValueError(
            f"HARD INVARIANT VIOLATION: Users in test_warm with <= {threshold_n} train interactions: {invalid_warm_users}"
        )

    # 4. Exact interaction leakage checks
    def df_to_tuple_set(df: pd.DataFrame) -> Set[Tuple[int, int, int, int]]:
        if df.empty:
            return set()
        return set(zip(df["user_id"], df["item_id"], df["rating"], df["timestamp"]))

    train_tuples = df_to_tuple_set(train_df)
    warm_test_tuples = df_to_tuple_set(test_warm_df)
    cold_test_tuples = df_to_tuple_set(test_cold_df)

    train_warm_leakage = train_tuples.intersection(warm_test_tuples)
    if train_warm_leakage:
        raise ValueError(f"HARD INVARIANT VIOLATION: Interaction leakage between train and test_warm ({len(train_warm_leakage)} rows)")

    train_cold_leakage = train_tuples.intersection(cold_test_tuples)
    if train_cold_leakage:
        raise ValueError(f"HARD INVARIANT VIOLATION: Interaction leakage between train and test_cold ({len(train_cold_leakage)} rows)")

    warm_cold_leakage = warm_test_tuples.intersection(cold_test_tuples)
    if warm_cold_leakage:
        raise ValueError(f"HARD INVARIANT VIOLATION: Interaction leakage between test_warm and test_cold ({len(warm_cold_leakage)} rows)")

    # 5. Row count completeness check
    total_split_rows = len(train_df) + len(test_warm_df) + len(test_cold_df)
    if total_split_rows != total_raw_count:
        raise ValueError(f"Row count mismatch: raw count={total_raw_count}, split sum={total_split_rows}")

    all_users = train_users.union(cold_users)
    all_items = set(train_df["item_id"]).union(set(test_warm_df["item_id"])).union(set(test_cold_df["item_id"]))

    summary = {
        "threshold_N": threshold_n,
        "total_interactions": total_split_rows,
        "total_users": len(all_users),
        "total_items": len(all_items),
        "cold_users_count": len(cold_users),
        "warm_users_count": len(warm_test_users),
        "train_interactions": len(train_df),
        "test_warm_interactions": len(test_warm_df),
        "test_cold_interactions": len(test_cold_df),
        "cold_users_in_train": len(cold_in_train),
        "cold_users_in_warm_test": len(cold_in_warm_test),
        "warm_test_users_absent_from_train": len(warm_absent_from_train),
        "train_warm_interaction_leakage": len(train_warm_leakage),
        "invalid_warm_users_count": len(invalid_warm_users)
    }

    return summary


def split_dataset(
    raw_dir: Union[str, Path] = RAW_DATA_DIR,
    threshold_n: int = THRESHOLD_N,
    warm_test_fraction: float = WARM_TEST_FRACTION,
    cold_user_fraction: float = COLD_USER_FRACTION,
    random_seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute strict train/warm/cold partitioning over MovieLens interaction data."""
    raw_df = load_interactions(raw_dir)
    total_raw_count = len(raw_df)

    unique_users = sorted(raw_df["user_id"].unique())
    cold_users = select_cold_users(unique_users, cold_user_fraction, random_seed)

    cold_df = raw_df[raw_df["user_id"].isin(cold_users)].copy()
    remaining_df = raw_df[~raw_df["user_id"].isin(cold_users)].copy()

    train_list = []
    test_warm_list = []

    for _, user_group in remaining_df.groupby("user_id"):
        user_train, user_test = split_warm_user_interactions(
            user_group,
            warm_test_fraction=warm_test_fraction,
            threshold_n=threshold_n
        )
        train_list.append(user_train)
        if not user_test.empty:
            test_warm_list.append(user_test)

    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame(columns=REQUIRED_COLUMNS)
    test_warm_df = pd.concat(test_warm_list, ignore_index=True) if test_warm_list else pd.DataFrame(columns=REQUIRED_COLUMNS)
    test_cold_df = cold_df.reset_index(drop=True)

    # Sort deterministically
    train_df = train_df.sort_values(by=["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    test_warm_df = test_warm_df.sort_values(by=["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    test_cold_df = test_cold_df.sort_values(by=["user_id", "timestamp", "item_id"]).reset_index(drop=True)

    # Enforce exact column types
    for df in [train_df, test_warm_df, test_cold_df]:
        df["user_id"] = df["user_id"].astype(int)
        df["item_id"] = df["item_id"].astype(int)
        df["rating"] = df["rating"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)

    # Run hard invariant checks
    validate_split(train_df, test_warm_df, test_cold_df, threshold_n=threshold_n, total_raw_count=total_raw_count)

    return train_df, test_warm_df, test_cold_df


def save_splits(
    raw_dir: Union[str, Path] = RAW_DATA_DIR,
    processed_dir: Union[str, Path] = PROCESSED_DATA_DIR,
    threshold_n: int = THRESHOLD_N,
    warm_test_fraction: float = WARM_TEST_FRACTION,
    cold_user_fraction: float = COLD_USER_FRACTION,
    random_seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Execute dataset split, validate invariants, and save CSV files to processed directory."""
    train_df, test_warm_df, test_cold_df = split_dataset(
        raw_dir=raw_dir,
        threshold_n=threshold_n,
        warm_test_fraction=warm_test_fraction,
        cold_user_fraction=cold_user_fraction,
        random_seed=random_seed
    )

    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_warm_path = out_dir / "test_warm.csv"
    test_cold_path = out_dir / "test_cold.csv"

    train_df[REQUIRED_COLUMNS].to_csv(train_path, index=False)
    test_warm_df[REQUIRED_COLUMNS].to_csv(test_warm_path, index=False)
    test_cold_df[REQUIRED_COLUMNS].to_csv(test_cold_path, index=False)

    summary = validate_split(train_df, test_warm_df, test_cold_df, threshold_n=threshold_n, total_raw_count=len(train_df)+len(test_warm_df)+len(test_cold_df))

    print(f"Data splits saved to '{out_dir}':")
    print(f"  train.csv: {len(train_df)} rows")
    print(f"  test_warm.csv: {len(test_warm_df)} rows")
    print(f"  test_cold.csv: {len(test_cold_df)} rows")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MovieLens strict train/warm/cold dataset splitting pipeline.")
    parser.add_argument("--threshold_n", type=int, default=THRESHOLD_N, help="Warm user interaction threshold N")
    parser.add_argument("--warm_test_fraction", type=float, default=WARM_TEST_FRACTION, help="Warm test holdout fraction")
    parser.add_argument("--cold_user_fraction", type=float, default=COLD_USER_FRACTION, help="Cold user holdout fraction")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")

    args = parser.parse_args()

    save_splits(
        threshold_n=args.threshold_n,
        warm_test_fraction=args.warm_test_fraction,
        cold_user_fraction=args.cold_user_fraction,
        random_seed=args.seed
    )
