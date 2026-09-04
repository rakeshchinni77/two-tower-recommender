"""Utility helper functions for seed reproducibility and environment setup."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across standard library, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
