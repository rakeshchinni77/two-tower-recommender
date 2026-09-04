"""PyTorch Two-Tower candidate retrieval neural network architecture."""

from typing import Union
import torch
import torch.nn as nn


class TwoTower(nn.Module):
    """Two-Tower retrieval model mapping user and item features into a shared dense embedding space.

    Internal ID Contract:
        0 = OOV / UNKNOWN / Padding bucket (reserved with padding_idx=0)
        1..num_users = valid internal user indices
        1..num_items = valid internal item indices
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Embedding tables with capacity for OOV index 0 plus 1..N valid IDs
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # Initialize embeddings with small normal variance (std=0.01)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        # Explicitly zero out index 0 OOV/padding weights
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0.0)
            self.item_embedding.weight[0].fill_(0.0)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Computes paired dot product similarity scores for user and item ID batches.

        Args:
            user_ids: Tensor of shape (batch_size,)
            item_ids: Tensor of shape (batch_size,)

        Returns:
            Tensor of shape (batch_size,) containing dot product scores.
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        return torch.sum(u_emb * i_emb, dim=-1)

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Fetch dense user embedding vectors for given user IDs."""
        return self.user_embedding(user_ids)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Alias for get_user_embedding."""
        return self.get_user_embedding(user_ids)

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Fetch dense item embedding vectors for given item IDs."""
        return self.item_embedding(item_ids)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Alias for get_item_embedding."""
        return self.get_item_embedding(item_ids)

    def score_all_items(self, user_id: Union[int, torch.Tensor]) -> torch.Tensor:
        """Compute dot product similarity scores between a single user and all valid items.

        Excludes the OOV item row at index 0 from candidate scores.

        Args:
            user_id: Single internal user integer ID or 1D long Tensor.

        Returns:
            Tensor of shape (num_items,) containing scores for valid items 1..num_items.
        """
        if isinstance(user_id, int):
            u_tensor = torch.tensor([user_id], dtype=torch.long, device=self.user_embedding.weight.device)
        else:
            u_tensor = user_id if user_id.ndim == 1 else user_id.squeeze()
            if u_tensor.ndim == 0:
                u_tensor = u_tensor.unsqueeze(0)

        u_emb = self.get_user_embedding(u_tensor)  # Shape: (1, embedding_dim)

        # Extract valid items matrix excluding OOV row index 0
        valid_item_embs = self.item_embedding.weight[1:]  # Shape: (num_items, embedding_dim)

        scores = torch.matmul(u_emb, valid_item_embs.T).squeeze(0)  # Shape: (num_items,)
        return scores
