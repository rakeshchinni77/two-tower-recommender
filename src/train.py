"""PyTorch Two-Tower model training pipeline using In-Batch Negatives loss."""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    EMBEDDING_DIM,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    RANDOM_SEED
)
from src.utils import set_seed
from models.two_tower import TwoTower


class InteractionDataset(Dataset):
    """PyTorch Dataset wrapping user-item interaction pairs."""

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray) -> None:
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx]


def load_training_data(processed_dir: Path = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, int, int]:
    """Load train.csv and save/load mapping files to ensure strict ID contract."""
    train_path = processed_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset missing at '{train_path}'")

    train_df = pd.read_csv(train_path)
    if train_df.empty:
        raise ValueError(f"Training dataset at '{train_path}' is empty")

    required_cols = ["user_id", "item_id", "rating", "timestamp"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column '{col}' in training dataset")

    # Load items metadata to obtain maximum total item count (1682)
    items_csv_path = processed_dir / "items.csv"
    if items_csv_path.exists():
        items_df = pd.read_csv(items_csv_path)
        num_items = int(items_df["item_id"].max())
    else:
        num_items = int(train_df["item_id"].max())

    num_users = int(train_df["user_id"].max())

    # Create and persist mapping files to data/processed
    user_mapping = {str(u): int(u) for u in range(1, num_users + 1)}
    item_mapping = {str(i): int(i) for i in range(1, num_items + 1)}

    user_map_path = processed_dir / "user_mapping.json"
    item_map_path = processed_dir / "item_mapping.json"

    with open(user_map_path, "w", encoding="utf-8") as f:
        json.dump(user_mapping, f, indent=2)

    with open(item_map_path, "w", encoding="utf-8") as f:
        json.dump(item_mapping, f, indent=2)

    return train_df, num_users, num_items


def create_dataloader(
    train_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader returning mapped internal user and item ID batches."""
    dataset = InteractionDataset(
        user_ids=train_df["user_id"].values,
        item_ids=train_df["item_id"].values
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )


def train_one_epoch(
    model: TwoTower,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    log_first_batch: bool = False
) -> float:
    """Train TwoTower model for one epoch using in-batch negative contrastive loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (user_ids, item_ids) in enumerate(dataloader):
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        batch_dim = user_ids.size(0)

        # 1. Obtain user and item embedding representations
        user_embs = model.get_user_embedding(user_ids)  # Shape: (B, D)
        item_embs = model.get_item_embedding(item_ids)  # Shape: (B, D)

        # 2. Compute in-batch similarity score matrix (B x B)
        scores = torch.matmul(user_embs, item_embs.T)  # Shape: (B, B)

        # 3. Targets: diagonal indices [0, 1, ..., B-1] represent positive pairs
        targets = torch.arange(batch_dim, device=device, dtype=torch.long)

        # Log structure verification on first batch of first epoch
        if log_first_batch and batch_idx == 0:
            print(f"Batch size: {batch_dim}")
            print(f"Score matrix shape: {scores.shape}")
            print(f"Target shape: {targets.shape}")

        # 4. Compute CrossEntropyLoss over score matrix against diagonal targets
        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(
    model: TwoTower,
    checkpoint_path: Path = ARTIFACTS_DIR / "two_tower.pt",
    config: Optional[Dict[str, Any]] = None
) -> Path:
    """Save structured PyTorch model checkpoint to artifacts/two_tower.pt."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "embedding_dim": model.embedding_dim,
        "num_users": model.num_users,
        "num_items": model.num_items,
        "config": config or {}
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> TwoTower:
    """Load model checkpoint into a fresh TwoTower instance and set to eval mode."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file missing at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TwoTower(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_smoke_test(model: TwoTower, device: torch.device) -> None:
    """Perform normal inference and OOV user index 0 verification tests."""
    model.eval()
    with torch.no_grad():
        # Normal inference check
        u_valid = torch.tensor([1], dtype=torch.long, device=device)
        i_valid = torch.tensor([10], dtype=torch.long, device=device)
        valid_score = model(u_valid, i_valid)

        assert valid_score.shape == (1,)
        assert torch.isfinite(valid_score).all()

        # OOV user index 0 check
        u_oov = torch.tensor([0], dtype=torch.long, device=device)
        oov_score = model(u_oov, i_valid)

        assert oov_score.shape == (1,)
        assert torch.isfinite(oov_score).all()
        assert oov_score.item() == 0.0

        # All-item scoring check
        scores_all = model.score_all_items(user_id=1)
        assert scores_all.shape == (model.num_items,)
        assert torch.isfinite(scores_all).all()

    print("Inference smoke test passed!")
    print(f"  Sample user 1 + item 10 score: {valid_score.item():.4f}")
    print(f"  OOV user 0 + item 10 score: {oov_score.item():.4f}")


def train_pipeline(
    processed_dir: Path = PROCESSED_DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    embedding_dim: int = EMBEDDING_DIM,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Execute complete Two-Tower in-batch negative training pipeline."""
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training data and IDs
    train_df, num_users, num_items = load_training_data(processed_dir)
    print(f"Loaded train interactions: {len(train_df)} rows")
    print(f"Number of users: {num_users}, Number of items: {num_items}")

    dataloader = create_dataloader(train_df, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TwoTower(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    config = {
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed
    }

    initial_loss = 0.0
    final_loss = 0.0

    print("Starting Two-Tower In-Batch Negatives Training...")
    for epoch in range(1, epochs + 1):
        log_first_batch = (epoch == 1)
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_first_batch=log_first_batch
        )
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f}")

        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

    # Save checkpoint
    ckpt_path = artifacts_dir / "two_tower.pt"
    save_checkpoint(model, checkpoint_path=ckpt_path, config=config)
    print(f"Saved checkpoint to '{ckpt_path}'")

    # Reload checkpoint and verify model integrity
    reloaded_model = load_checkpoint(ckpt_path, device=device)
    run_smoke_test(reloaded_model, device=device)

    summary = {
        "device": str(device),
        "train_interactions": len(train_df),
        "num_users": num_users,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "checkpoint_path": str(ckpt_path),
        "oov_test_passed": True
    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Two-Tower In-Batch Negative Training Pipeline")
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM, help="Embedding dimension (default: 64)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="In-batch negative batch size (default: 256)")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Optimizer learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs (default: 10)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")

    args = parser.parse_args()

    train_pipeline(
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed
    )
