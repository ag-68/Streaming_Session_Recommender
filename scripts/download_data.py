"""Download MovieLens 1M and check the raw file before any filtering."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessrec.data import assert_raw_file, read_ratings  # noqa: E402
from sessrec.paths import DATASET_DIR, RATINGS_FILE, RAW_DIR  # noqa: E402

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RATINGS_FILE.exists():
        assert_raw_file(read_ratings())
        print(f"already present: {DATASET_DIR}")
        return
    destination = RAW_DIR / "ml-1m.zip"
    print(f"downloading {URL}")
    response = httpx.get(URL, follow_redirects=True, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    with zipfile.ZipFile(destination) as archive:
        archive.extractall(RAW_DIR)
    assert_raw_file(read_ratings())
    print(f"ready: {DATASET_DIR}")


if __name__ == "__main__":
    download()
