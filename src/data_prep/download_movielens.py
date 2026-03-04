"""Download and extract the MovieLens ml-latest-small dataset."""
from __future__ import annotations

import io
import zipfile

import requests
from tqdm import tqdm

from src.common.config import RAW_DIR, ensure_dirs, get_recsys_config
from src.common.logging import get_logger, setup_logging

logger = get_logger(__name__)


def download_movielens() -> None:
    """Download ml-latest-small and extract to data/raw/."""
    ensure_dirs()
    cfg = get_recsys_config()
    url = cfg["data"]["movielens_url"]

    target = RAW_DIR / "ml-latest-small"
    if (target / "ratings.csv").exists():
        logger.info("MovieLens data already exists, skipping download.")
        return

    logger.info("Downloading MovieLens ml-latest-small", url=url)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            buf.write(chunk)
            pbar.update(len(chunk))

    logger.info("Extracting archive")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(RAW_DIR)

    logger.info("MovieLens data ready", path=str(target))


if __name__ == "__main__":
    setup_logging()
    download_movielens()
