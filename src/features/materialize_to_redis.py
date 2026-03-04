"""Materialize offline features to Redis for online serving."""
from __future__ import annotations

import json
import os

import redis

from src.common.config import OFFLINE_FEATURES_DIR, ensure_dirs, get_features_config
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet

logger = get_logger(__name__)


def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True,
    )


def materialize_features() -> None:
    """Write offline features to Redis as JSON."""
    ensure_dirs()
    cfg = get_features_config()
    batch_size = cfg.get("materialization", {}).get("batch_size", 500)
    user_prefix = cfg["redis_keys"]["user_features_prefix"]
    item_prefix = cfg["redis_keys"]["item_features_prefix"]

    r = get_redis_client()

    # User features
    user_df = load_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    logger.info("Materializing user features", n_users=len(user_df))

    pipe = r.pipeline()
    for i, row in enumerate(user_df.itertuples()):
        key = f"{user_prefix}{row.user_id}"
        val = json.dumps({
            "user_watch_count_30d": int(row.user_watch_count_30d),
            "user_genre_affinity": row.user_genre_affinity,
            "user_mean_rating": float(row.user_mean_rating) if row.user_mean_rating == row.user_mean_rating else 0.0,
            "user_recency_mean_days": float(row.user_recency_mean_days) if row.user_recency_mean_days == row.user_recency_mean_days else 0.0,
        })
        pipe.set(key, val)
        if (i + 1) % batch_size == 0:
            pipe.execute()
            pipe = r.pipeline()

    pipe.execute()
    logger.info("User features materialized")

    # Item features
    item_df = load_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")
    logger.info("Materializing item features", n_items=len(item_df))

    pipe = r.pipeline()
    for i, row in enumerate(item_df.itertuples()):
        key = f"{item_prefix}{row.item_id}"
        val = json.dumps({
            "item_watch_count": int(row.item_watch_count),
            "genres": row.genres if isinstance(row.genres, str) else "",
            "genres_multihot": row.genres_multihot,
            "item_mean_rating": float(row.item_mean_rating) if row.item_mean_rating == row.item_mean_rating else 0.0,
            "title": row.title if isinstance(row.title, str) else "",
        })
        pipe.set(key, val)
        if (i + 1) % batch_size == 0:
            pipe.execute()
            pipe = r.pipeline()

    pipe.execute()
    logger.info("Item features materialized")


if __name__ == "__main__":
    setup_logging()
    materialize_features()
