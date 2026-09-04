"""Unit tests for Phase 6 PyTorch Two-Tower training pipeline and checkpointing."""

import pytest
import torch
import pandas as pd
from pathlib import Path

from src.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR, EMBEDDING_DIM, BATCH_SIZE
from src.train import (
    load_training_data,
    create_dataloader,
    save_checkpoint,
    load_checkpoint,
    run_smoke_test
)
from models.two_tower import TwoTower


def test_load_training_data_uses_only_train_csv():
    train_df, num_users, num_items = load_training_data(PROCESSED_DATA_DIR)

    assert not train_df.empty
    assert len(train_df) == 73727
    assert num_users == 943
    assert num_items == 1682

    # Check mapping files created
    assert (PROCESSED_DATA_DIR / "user_mapping.json").exists()
    assert (PROCESSED_DATA_DIR / "item_mapping.json").exists()


def test_dataloader_batch_dimensions():
    train_df, _, _ = load_training_data(PROCESSED_DATA_DIR)
    dataloader = create_dataloader(train_df, batch_size=256, shuffle=False, drop_last=True)

    user_ids, item_ids = next(iter(dataloader))

    assert user_ids.shape == (256,)
    assert item_ids.shape == (256,)
    assert user_ids.dtype == torch.long
    assert item_ids.dtype == torch.long


def test_checkpoint_saving_and_loading(tmp_path):
    model = TwoTower(num_users=10, num_items=20, embedding_dim=16)
    ckpt_path = tmp_path / "two_tower.pt"

    config = {"embedding_dim": 16, "batch_size": 32}
    save_checkpoint(model, checkpoint_path=ckpt_path, config=config)

    assert ckpt_path.exists()

    device = torch.device("cpu")
    reloaded_model = load_checkpoint(ckpt_path, device=device)

    assert reloaded_model.num_users == 10
    assert reloaded_model.num_items == 20
    assert reloaded_model.embedding_dim == 16


def test_trained_checkpoint_artifacts_integrity():
    ckpt_path = ARTIFACTS_DIR / "two_tower.pt"
    assert ckpt_path.exists(), f"artifacts/two_tower.pt missing at {ckpt_path}"

    device = torch.device("cpu")
    model = load_checkpoint(ckpt_path, device=device)

    assert model.embedding_dim == 64
    assert model.num_users == 943
    assert model.num_items == 1682

    # Smoke test on loaded artifact
    run_smoke_test(model, device=device)
