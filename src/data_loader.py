"""Data loader and integrity validation module for MovieLens 100K dataset."""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]


def load_interactions(raw_dir: Union[str, Path] = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and parse raw user-item interactions from u.data.

    Preserves original MovieLens user_id and item_id integers.
    """
    data_path = Path(raw_dir) / "u.data"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing interaction raw file at '{data_path}'")

    cols = ["user_id", "item_id", "rating", "timestamp"]
    try:
        df = pd.read_csv(data_path, sep="\t", names=cols, engine="python")
    except Exception as e:
        raise ValueError(f"Failed to parse u.data file at '{data_path}': {e}")

    if df.empty:
        raise ValueError(f"Loaded u.data interaction file is empty: '{data_path}'")

    # Enforce strict data types and check nulls
    for col in cols:
        if df[col].isnull().any():
            raise ValueError(f"Null values detected in column '{col}' of interaction data")

    try:
        df["user_id"] = df["user_id"].astype(int)
        df["item_id"] = df["item_id"].astype(int)
        df["rating"] = df["rating"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)
    except Exception as e:
        raise ValueError(f"Data type conversion error in interaction data: {e}")

    # Validate value constraints
    if not df["rating"].between(1, 5).all():
        invalid_ratings = df[~df["rating"].between(1, 5)]["rating"].unique()
        raise ValueError(f"Invalid rating values found in u.data: {invalid_ratings}")

    if (df["timestamp"] <= 0).any():
        raise ValueError("Non-positive timestamp values detected in u.data")

    return df


def load_items(raw_dir: Union[str, Path] = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and parse raw movie metadata from u.item using legacy encoding.

    Preserves original MovieLens item_id integers and combines genre indicators.
    """
    item_path = Path(raw_dir) / "u.item"
    if not item_path.exists():
        raise FileNotFoundError(f"Missing movie metadata raw file at '{item_path}'")

    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLUMNS

    # Legacy encoding strategy handling latin-1 / iso-8859-1 encoding explicitly
    try:
        items_df = pd.read_csv(
            item_path,
            sep="|",
            names=cols,
            encoding="latin-1",
            engine="python"
        )
    except Exception as e:
        raise ValueError(f"Failed to parse u.item file at '{item_path}': {e}")

    if items_df.empty:
        raise ValueError(f"Loaded u.item metadata file is empty: '{item_path}'")

    if items_df["item_id"].isnull().any() or items_df["title"].isnull().any():
        raise ValueError("Null values detected in item_id or title of movie metadata")

    try:
        items_df["item_id"] = items_df["item_id"].astype(int)
    except Exception as e:
        raise ValueError(f"Data type conversion error in movie item_id: {e}")

    # Convert binary genre columns into a clean pipe-separated string
    def format_genres(row: pd.Series) -> str:
        active_genres = [g for g in GENRE_COLUMNS if row[g] == 1]
        return "|".join(active_genres) if active_genres else "Unknown"

    items_df["genres"] = items_df.apply(format_genres, axis=1)

    return items_df[["item_id", "title", "genres", "release_date", "imdb_url"]]


def load_users(raw_dir: Union[str, Path] = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and parse raw user metadata from u.user."""
    user_path = Path(raw_dir) / "u.user"
    if not user_path.exists():
        raise FileNotFoundError(f"Missing user raw file at '{user_path}'")

    cols = ["user_id", "age", "gender", "occupation", "zip_code"]
    try:
        df = pd.read_csv(user_path, sep="|", names=cols, engine="python")
    except Exception as e:
        raise ValueError(f"Failed to parse u.user file at '{user_path}': {e}")

    if df.empty:
        raise ValueError(f"Loaded u.user file is empty: '{user_path}'")

    df["user_id"] = df["user_id"].astype(int)
    return df


def validate_dataset(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Execute integrity validation and metadata join checks on loaded data."""
    if interactions_df.empty:
        raise ValueError("Interactions DataFrame is empty")
    if items_df.empty:
        raise ValueError("Item metadata DataFrame is empty")

    n_interactions = len(interactions_df)
    n_users = interactions_df["user_id"].nunique()
    n_items = interactions_df["item_id"].nunique()
    n_meta_items = len(items_df)

    min_rating = int(interactions_df["rating"].min())
    max_rating = int(interactions_df["rating"].max())
    unique_ratings = sorted(interactions_df["rating"].unique().tolist())
    min_ts = int(interactions_df["timestamp"].min())
    max_ts = int(interactions_df["timestamp"].max())

    # Metadata Join Integrity Check
    interaction_item_ids = set(interactions_df["item_id"].unique())
    metadata_item_ids = set(items_df["item_id"].unique())

    unresolved_items = interaction_item_ids - metadata_item_ids
    if unresolved_items:
        raise ValueError(
            f"Metadata join failure: {len(unresolved_items)} interaction item IDs not found in item metadata: {unresolved_items}"
        )

    # Duplicate check (report without dropping)
    duplicates = interactions_df.duplicated(subset=["user_id", "item_id"]).sum()

    stats = {
        "interaction_count": n_interactions,
        "unique_users": n_users,
        "unique_items_in_interactions": n_items,
        "unique_items_in_metadata": n_meta_items,
        "rating_range": (min_rating, max_rating),
        "unique_ratings": unique_ratings,
        "timestamp_range": (min_ts, max_ts),
        "duplicate_interaction_count": int(duplicates),
        "join_validation_passed": True
    }

    return stats


def load_dataset(raw_dir: Union[str, Path] = RAW_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function loading interactions and items with integrity validation."""
    interactions = load_interactions(raw_dir)
    items = load_items(raw_dir)
    users = load_users(raw_dir) if (Path(raw_dir) / "u.user").exists() else None

    # Validate dataset integrity
    validate_dataset(interactions, items, users)

    return interactions, items


if __name__ == "__main__":
    print("Loading MovieLens 100K raw dataset...")
    interactions, items = load_dataset()
    stats = validate_dataset(interactions, items)
    print("Dataset validation successful!")
    for k, v in stats.items():
        print(f"  {k}: {v}")
