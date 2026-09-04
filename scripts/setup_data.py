"""Data acquisition script for MovieLens 100K dataset."""

import os
import sys
import ssl
import zipfile
import urllib.request
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
REQUIRED_RAW_FILES = ["u.data", "u.item", "u.user", "README"]


def download_and_extract_movielens(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Download ml-100k.zip and extract required files to data/raw.

    Safe to run repeatedly; skips download if required raw files exist.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if all required raw files are already present and non-empty
    all_exist = all((raw_dir / fname).exists() and (raw_dir / fname).stat().st_size > 0 for fname in REQUIRED_RAW_FILES)
    if all_exist:
        print(f"MovieLens 100K dataset raw files already exist in '{raw_dir}'. Skipping download.")
        return

    zip_path = raw_dir / "ml-100k.zip"
    print(f"Downloading MovieLens 100K from '{MOVIELENS_100K_URL}'...")

    # Handle SSL context for public dataset server
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(
        MOVIELENS_100K_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, context=ssl_context) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    print(f"Extracting raw files to '{raw_dir}'...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue
            # Extract required files into raw_dir directly
            if filename in REQUIRED_RAW_FILES or filename.startswith("u."):
                target_path = raw_dir / filename
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

    # Clean up zip file if extracted successfully
    if zip_path.exists():
        zip_path.unlink()

    # Final check of required files
    missing = [fname for fname in REQUIRED_RAW_FILES if not (raw_dir / fname).exists()]
    if missing:
        raise RuntimeError(f"Data acquisition failed; missing raw files in '{raw_dir}': {missing}")

    print("MovieLens 100K data acquisition completed successfully.")


if __name__ == "__main__":
    download_and_extract_movielens()
