"""Comprehensive unit test suite for PyTorch TwoTower retrieval model architecture."""

import pytest
import torch
import torch.optim as optim
from models.two_tower import TwoTower


def test_model_construction_and_embedding_dimensions():
    num_users = 100
    num_items = 200
    embedding_dim = 64

    model = TwoTower(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim)

    assert model.embedding_dim == 64
    assert model.num_users == 100
    assert model.num_items == 200

    # Capacity check: index 0 (OOV) + 1..N valid IDs = N + 1 rows
    assert model.user_embedding.weight.shape == (101, 64)
    assert model.item_embedding.weight.shape == (201, 64)


def test_embedding_table_capacity_and_padding_idx():
    model = TwoTower(num_users=50, num_items=80, embedding_dim=32)

    assert model.user_embedding.padding_idx == 0
    assert model.item_embedding.padding_idx == 0

    assert model.user_embedding.num_embeddings == 51
    assert model.item_embedding.num_embeddings == 81


def test_forward_pass_vectorized_batch():
    model = TwoTower(num_users=10, num_items=20, embedding_dim=16)

    user_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    item_ids = torch.tensor([5, 6, 7, 8], dtype=torch.long)

    scores = model(user_ids, item_ids)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()


def test_dot_product_numerical_correctness():
    model = TwoTower(num_users=5, num_items=5, embedding_dim=2)

    # Manually set weights to verify mathematical correctness
    with torch.no_grad():
        model.user_embedding.weight[1] = torch.tensor([1.0, 2.0])
        model.item_embedding.weight[3] = torch.tensor([3.0, 4.0])

    user_ids = torch.tensor([1], dtype=torch.long)
    item_ids = torch.tensor([3], dtype=torch.long)

    score = model(user_ids, item_ids)

    # Expected: 1*3 + 2*4 = 11.0
    assert abs(score.item() - 11.0) < 1e-5


def test_oov_user_handling():
    model = TwoTower(num_users=10, num_items=10, embedding_dim=16)

    user_ids = torch.tensor([0], dtype=torch.long)
    item_ids = torch.tensor([5], dtype=torch.long)

    scores = model(user_ids, item_ids)

    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()
    # OOV user embedding is 0 vector, so dot product with any item is 0.0
    assert scores.item() == 0.0


def test_oov_item_handling():
    model = TwoTower(num_users=10, num_items=10, embedding_dim=16)

    user_ids = torch.tensor([3], dtype=torch.long)
    item_ids = torch.tensor([0], dtype=torch.long)

    scores = model(user_ids, item_ids)

    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()
    assert scores.item() == 0.0


def test_multiple_oov_ids_batch():
    model = TwoTower(num_users=10, num_items=10, embedding_dim=16)

    user_ids = torch.tensor([0, 1, 0, 2], dtype=torch.long)
    item_ids = torch.tensor([1, 0, 3, 0], dtype=torch.long)

    scores = model(user_ids, item_ids)

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()


def test_padding_oov_row_is_zero_initialized():
    model = TwoTower(num_users=10, num_items=10, embedding_dim=16)

    user_oov = model.user_embedding.weight[0]
    item_oov = model.item_embedding.weight[0]

    assert (user_oov == 0.0).all()
    assert (item_oov == 0.0).all()


def test_score_all_items_excludes_oov():
    num_items = 25
    model = TwoTower(num_users=10, num_items=num_items, embedding_dim=16)

    scores = model.score_all_items(user_id=2)

    assert isinstance(scores, torch.Tensor)
    assert scores.shape == (num_items,)
    assert torch.isfinite(scores).all()

    # Scores count must equal num_items (valid items 1..25, OOV index 0 excluded)
    assert len(scores) == num_items


def test_gradient_safety_for_oov_padding_row():
    model = TwoTower(num_users=5, num_items=5, embedding_dim=4)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Pass batch including OOV index 0
    user_ids = torch.tensor([0, 1], dtype=torch.long)
    item_ids = torch.tensor([1, 0], dtype=torch.long)

    scores = model(user_ids, item_ids)
    loss = scores.sum()
    loss.backward()

    optimizer.step()

    # OOV index 0 row weights must remain zero after backpropagation and optimizer step
    assert (model.user_embedding.weight[0] == 0.0).all()
    assert (model.item_embedding.weight[0] == 0.0).all()
