"""Online feature client: fetches features from Redis with offline fallback."""
from __future__ import annotations

import json
import time

import pandas as pd

from src.common.config import OFFLINE_FEATURES_DIR
from src.common.logging import get_logger
from src.common.metrics import FEATURE_FETCH_LATENCY
from src.common.utils import load_parquet
from src.serving.cache import CacheClient

logger = get_logger(__name__)


class FeatureClient:
    """Fetches user/item features from cache (Redis) with offline parquet fallback."""

    def __init__(self, cache: CacheClient) -> None:
        self._cache = cache

        # Load offline features as fallback
        try:
            uf = load_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
            self._offline_users: dict[int, dict] = {}
            for row in uf.itertuples():
                self._offline_users[row.user_id] = {
                    "user_watch_count_30d": int(row.user_watch_count_30d),
                    "user_genre_affinity": row.user_genre_affinity,
                    "user_mean_rating": float(row.user_mean_rating) if row.user_mean_rating == row.user_mean_rating else 0.0,
                    "user_recency_mean_days": float(row.user_recency_mean_days) if row.user_recency_mean_days == row.user_recency_mean_days else 0.0,
                }
        except Exception:
            self._offline_users = {}

        try:
            itf = load_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")
            self._offline_items: dict[int, dict] = {}
            for row in itf.itertuples():
                self._offline_items[row.item_id] = {
                    "item_watch_count": int(row.item_watch_count),
                    "genres": row.genres if isinstance(row.genres, str) else "",
                    "genres_multihot": row.genres_multihot if isinstance(row.genres_multihot, str) else "[]",
                    "item_mean_rating": float(row.item_mean_rating) if row.item_mean_rating == row.item_mean_rating else 0.0,
                    "title": row.title if isinstance(row.title, str) else "",
                }
        except Exception:
            self._offline_items = {}

        logger.info(
            "FeatureClient initialized",
            offline_users=len(self._offline_users),
            offline_items=len(self._offline_items),
        )

    def get_user_features(self, user_id: int) -> dict | None:
        """Get user features from cache or offline fallback."""
        start = time.monotonic()
        try:
            # Try cache (Redis)
            feat = self._cache.get_user_features(user_id)
            if feat is not None:
                return feat

            # Fallback to offline
            return self._offline_users.get(user_id)
        finally:
            FEATURE_FETCH_LATENCY.observe(time.monotonic() - start)

    def get_item_features(self, item_id: int) -> dict | None:
        """Get item features from cache or offline fallback."""
        feat = self._cache.get_item_features(item_id)
        if feat is not None:
            return feat
        return self._offline_items.get(item_id)
