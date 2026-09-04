"""Popularity and Content-Similarity Baseline Recommenders."""

import sys
from pathlib import Path
from typing import List, Dict, Set, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, ITEMS_PATH, ITEM_EMBEDDINGS_PATH


class PopularityBaseline:
    """Popularity baseline recommender ranking items globally by interaction_count * average_rating."""

    def __init__(self) -> None:
        self.popular_items: List[int] = []
        self.item_scores: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityBaseline":
        """Compute popularity scores ONLY from training interactions:
        popularity_score = interaction_count * average_rating.
        """
        if train_df.empty:
            self.popular_items = []
            self.item_scores = {}
            return self

        stats = train_df.groupby("item_id").agg(
            interaction_count=("rating", "count"),
            average_rating=("rating", "mean")
        ).reset_index()

        stats["popularity_score"] = stats["interaction_count"] * stats["average_rating"]
        stats["popularity_score"] = stats["popularity_score"].fillna(0.0)

        # Sort descending by popularity_score, tie-break by item_id ascending
        stats = stats.sort_values(by=["popularity_score", "item_id"], ascending=[False, True])

        self.popular_items = [int(x) for x in stats["item_id"].tolist()]
        self.item_scores = {int(k): float(v) for k, v in zip(stats["item_id"], stats["popularity_score"])}
        return self

    def recommend(
        self,
        top_k: int = 10,
        exclude_items: Optional[Union[List[int], Set[int]]] = None,
        exclude_item_ids: Optional[Union[List[int], Set[int]]] = None
    ) -> List[int]:
        """Return global top-K popular item IDs, excluding specified items if provided."""
        exclude = set()
        if exclude_items is not None:
            exclude.update(exclude_items)
        if exclude_item_ids is not None:
            exclude.update(exclude_item_ids)

        recs = [int(item_id) for item_id in self.popular_items if int(item_id) not in exclude]
        return recs[:top_k]


# Alias for PopularityBaseline
PopularityRecommender = PopularityBaseline


class ContentBaseline:
    """Content similarity baseline recommender using metadata text embeddings and cosine similarity."""

    def __init__(
        self,
        item_embeddings: np.ndarray,
        item_ids: List[int],
        popularity_recommender: Optional[PopularityBaseline] = None
    ) -> None:
        if not isinstance(item_embeddings, np.ndarray) or item_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be a 2D NumPy ndarray")
        if len(item_ids) != item_embeddings.shape[0]:
            raise ValueError(f"item_ids length ({len(item_ids)}) must match embeddings count ({item_embeddings.shape[0]})")

        self.item_embeddings = item_embeddings.astype(np.float32)
        self.item_ids = [int(x) for x in item_ids]
        self.item_id_to_idx: Dict[int, int] = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.popularity_recommender = popularity_recommender

        # Precompute L2 norms for item embeddings to accelerate cosine similarity safely
        norms = np.linalg.norm(self.item_embeddings, axis=1)
        self.item_norms_safe = np.where(norms > 0, norms, 1e-10)

    @classmethod
    def load_from_processed(
        cls,
        processed_dir: Union[str, Path] = PROCESSED_DATA_DIR,
        popularity_recommender: Optional[PopularityBaseline] = None
    ) -> "ContentBaseline":
        """Factory method loading item_embeddings.npy and items.csv from data/processed."""
        processed_path = Path(processed_dir)
        items_csv_path = processed_path / "items.csv"
        npy_path = processed_path / "item_embeddings.npy"

        if not items_csv_path.exists():
            raise FileNotFoundError(f"Missing items metadata file at {items_csv_path}")
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing item embeddings file at {npy_path}")

        items_df = pd.read_csv(items_csv_path)
        embeddings = np.load(npy_path)
        item_ids = [int(x) for x in items_df["item_id"].tolist()]
        return cls(item_embeddings=embeddings, item_ids=item_ids, popularity_recommender=popularity_recommender)

    def recommend(
        self,
        history_item_ids: Optional[List[int]] = None,
        top_k: int = 10,
        exclude_items: Optional[Union[List[int], Set[int]]] = None,
        history: Optional[List[int]] = None
    ) -> List[int]:
        """Recommend items based on user history content centroid via cosine similarity.

        - If history is empty / zero valid items, cascades to popularity_recommender if available,
          or returns empty list if no popularity_recommender is attached.
        - Excludes historical items and exclude_items from recommendations.
        """
        # Support history argument aliases
        hist = history_item_ids if history_item_ids is not None else (history if history is not None else [])

        # Build combined exclusion set
        exclude_set = set(hist)
        if exclude_items is not None:
            exclude_set.update(exclude_items)

        # Filter valid unique history item IDs
        valid_indices = []
        seen_history = set()
        for item_id in hist:
            try:
                item_id_int = int(item_id)
            except (ValueError, TypeError):
                continue
            if item_id_int in self.item_id_to_idx and item_id_int not in seen_history:
                valid_indices.append(self.item_id_to_idx[item_id_int])
                seen_history.add(item_id_int)

        # Zero-history fallback
        if not valid_indices:
            if self.popularity_recommender is not None:
                return self.popularity_recommender.recommend(top_k=top_k, exclude_items=exclude_set)
            return []

        # Calculate mean content vector across user's history item embeddings
        history_embs = self.item_embeddings[valid_indices]
        user_profile = np.mean(history_embs, axis=0, keepdims=True)

        user_norm = np.linalg.norm(user_profile)
        user_norm_safe = user_norm if user_norm > 0 else 1e-10

        # Calculate cosine similarity using sklearn or numpy dot products
        dot_products = np.dot(self.item_embeddings, user_profile.T).squeeze()
        sims = dot_products / (self.item_norms_safe * user_norm_safe)
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

        # Rank items by similarity descending
        sorted_indices = np.argsort(-sims)

        recs: List[int] = []
        for idx in sorted_indices:
            candidate_id = self.item_ids[idx]
            if candidate_id not in exclude_set:
                recs.append(candidate_id)
            if len(recs) == top_k:
                break

        return recs


# Alias for ContentBaseline
ContentRecommender = ContentBaseline
