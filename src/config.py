"""Configuration management loading environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = BASE_DIR / os.getenv("ARTIFACTS_DIR", "artifacts")
RESULTS_DIR = BASE_DIR / os.getenv("RESULTS_DIR", "results")

THRESHOLD_N = int(os.getenv("THRESHOLD_N", "10"))
WARM_TEST_FRACTION = float(os.getenv("WARM_TEST_FRACTION", "0.20"))
COLD_USER_FRACTION = float(os.getenv("COLD_USER_FRACTION", "0.10"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_ID = EMBEDDING_MODEL

ITEMS_PATH = PROCESSED_DATA_DIR / "items.csv"
ITEM_EMBEDDINGS_PATH = PROCESSED_DATA_DIR / "item_embeddings.npy"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
