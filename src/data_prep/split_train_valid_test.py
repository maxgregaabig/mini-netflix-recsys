"""Split events into train/valid/test by time per user."""
from __future__ import annotations

import pandas as pd

from src.common.config import PROCESSED_DIR, ensure_dirs, get_recsys_config
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet, save_parquet

logger = get_logger(__name__)


def split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split: oldest events → train, middle → valid, newest → test."""
    ensure_dirs()
    cfg = get_recsys_config()["data"]

    events = load_parquet(PROCESSED_DIR / "events.parquet")

    train_ratio = cfg["train_ratio"]
    valid_ratio = cfg["valid_ratio"]

    train_parts, valid_parts, test_parts = [], [], []

    for _, user_events in events.groupby("user_id"):
        user_events = user_events.sort_values("timestamp")
        n = len(user_events)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))

        train_parts.append(user_events.iloc[:train_end])
        valid_parts.append(user_events.iloc[train_end:valid_end])
        test_parts.append(user_events.iloc[valid_end:])

    train = pd.concat(train_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    save_parquet(train, PROCESSED_DIR / "train.parquet")
    save_parquet(valid, PROCESSED_DIR / "valid.parquet")
    save_parquet(test, PROCESSED_DIR / "test.parquet")

    logger.info(
        "Data split complete",
        train=len(train),
        valid=len(valid),
        test=len(test),
    )
    return train, valid, test


if __name__ == "__main__":
    setup_logging()
    split_data()
