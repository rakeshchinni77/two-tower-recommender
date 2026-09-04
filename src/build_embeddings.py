"""Content embedding generation module using SentenceTransformers."""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ITEMS_PATH,
    ITEM_EMBEDDINGS_PATH,
    EMBEDDING_MODEL_ID
)
from src.data_loader import load_items


def format_item_text(row: pd.Series) -> str:
    """Format title and genres into structured text for content encoding."""
    title = str(row["title"]).strip()
    genres = str(row["genres"]).strip()
    if genres and genres != "Unknown":
        formatted_genres = genres.replace("|", " | ")
        return f"{title}: {formatted_genres}"
    return f"{title}: "


def generate_items_df(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load items metadata and generate required text column."""
    items_raw = load_items(raw_dir)
    items_df = items_raw.copy()

    items_df["text"] = items_df.apply(format_item_text, axis=1)

    required_cols = ["item_id", "title", "genres", "text"]
    items_df = items_df[required_cols]

    # Validate item metadata integrity
    if len(items_df) != 1682:
        raise ValueError(f"Expected 1682 items in MovieLens 100K, got {len(items_df)}")
    if items_df["item_id"].nunique() != 1682:
        raise ValueError("Item IDs in items.csv are not unique")

    return items_df


def validate_embeddings(embeddings: np.ndarray, expected_count: int = 1682, expected_dim: int = 384) -> None:
    """Validate embedding matrix structure, dimensions, data type, and numerical safety."""
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Embeddings output must be a NumPy ndarray")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got {embeddings.ndim}D")
    if embeddings.shape[0] != expected_count:
        raise ValueError(f"Expected {expected_count} rows, got {embeddings.shape[0]}")
    if embeddings.shape[1] != expected_dim:
        raise ValueError(f"Expected embedding dimension {expected_dim}, got {embeddings.shape[1]}")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise TypeError(f"Embedding matrix dtype must be floating point, got {embeddings.dtype}")
    if np.isnan(embeddings).any():
        raise ValueError("Embedding matrix contains NaN values")
    if np.isinf(embeddings).any():
        raise ValueError("Embedding matrix contains Inf values")
    if np.all(embeddings == 0):
        raise ValueError("Embedding matrix is entirely zero")


def build_item_embeddings(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    model_id: str = EMBEDDING_MODEL_ID
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate items.csv and item_embeddings.npy using local SentenceTransformer."""
    print("Loading MovieLens metadata...")
    items_df = generate_items_df(raw_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    items_csv_path = processed_dir / "items.csv"
    items_df.to_csv(items_csv_path, index=False)

    print(f"Items: {len(items_df)}")
    print(f"Embedding model: {model_id}")

    try:
        model = SentenceTransformer(model_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load local SentenceTransformer model '{model_id}': {e}")

    texts = items_df["text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    validate_embeddings(embeddings, expected_count=len(items_df), expected_dim=embeddings.shape[1])

    npy_path = processed_dir / "item_embeddings.npy"
    np.save(npy_path, embeddings)

    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Saved: {items_csv_path}")
    print(f"Saved: {npy_path}")
    print(f"Embedding shape: {embeddings.shape}")

    return items_df, embeddings


if __name__ == "__main__":
    build_item_embeddings()
