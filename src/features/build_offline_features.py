"""Build offline feature tables from events data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import OFFLINE_FEATURES_DIR, PROCESSED_DIR, ensure_dirs
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet, save_parquet
from src.features.feature_defs import ALL_GENRES, GENRE_TO_IDX

logger = get_logger(__name__)


def build_user_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build user-level features from events."""
    max_ts = events["timestamp"].max()

    # Watch count in last 30 days (synthetic window)
    cutoff = max_ts - pd.Timedelta(days=30)
    recent = events[events["timestamp"] >= cutoff]
    watch_30d = recent.groupby("user_id").size().rename("user_watch_count_30d")

    # Genre affinity: normalized genre counts
    def compute_genre_affinity(group: pd.DataFrame) -> str:
        genre_counts: dict[str, float] = {}
        for genres_str in group["genres"].dropna():
            for g in genres_str.split("|"):
                g = g.strip()
                if g in GENRE_TO_IDX:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        total = sum(genre_counts.values()) or 1
        return json.dumps({g: round(c / total, 4) for g, c in genre_counts.items()})

    genre_aff = events.groupby("user_id").apply(compute_genre_affinity, include_groups=False)
    genre_aff = genre_aff.rename("user_genre_affinity")

    # Mean rating
    mean_rating = events.groupby("user_id")["rating"].mean().rename("user_mean_rating").round(4)

    # Recency: mean days since recent watches
    days_since = (max_ts - events["timestamp"]).dt.total_seconds() / 86400
    events = events.copy()
    events["days_since"] = days_since
    recency = events.groupby("user_id")["days_since"].mean().rename("user_recency_mean_days").round(2)

    user_df = pd.DataFrame({"user_id": events["user_id"].unique()})
    user_df = user_df.merge(watch_30d, on="user_id", how="left").fillna({"user_watch_count_30d": 0})
    user_df = user_df.merge(genre_aff, on="user_id", how="left")
    user_df = user_df.merge(mean_rating, on="user_id", how="left")
    user_df = user_df.merge(recency, on="user_id", how="left")

    user_df["user_watch_count_30d"] = user_df["user_watch_count_30d"].astype(int)
    return user_df


def build_item_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build item-level features from events."""
    movies = load_parquet(PROCESSED_DIR / "movies.parquet")

    # Watch count (popularity)
    watch_count = events.groupby("item_id").size().rename("item_watch_count")

    # Mean rating
    mean_rating = events.groupby("item_id")["rating"].mean().rename("item_mean_rating").round(4)

    item_df = movies[["item_id", "title", "genres"]].copy()
    item_df = item_df.merge(watch_count, on="item_id", how="left").fillna({"item_watch_count": 0})
    item_df = item_df.merge(mean_rating, on="item_id", how="left")
    item_df["item_watch_count"] = item_df["item_watch_count"].astype(int)

    # Multi-hot genre encoding
    def genres_to_multihot(genres_str: str) -> str:
        vec = [0] * len(ALL_GENRES)
        if pd.isna(genres_str):
            return json.dumps(vec)
        for g in genres_str.split("|"):
            g = g.strip()
            if g in GENRE_TO_IDX:
                vec[GENRE_TO_IDX[g]] = 1
        return json.dumps(vec)

    item_df["genres_multihot"] = item_df["genres"].apply(genres_to_multihot)
    return item_df


def build_all_features() -> None:
    """Build and save all offline feature tables."""
    ensure_dirs()

    train = load_parquet(PROCESSED_DIR / "train.parquet")
    logger.info("Building user features", n_events=len(train))

    user_features = build_user_features(train)
    save_parquet(user_features, OFFLINE_FEATURES_DIR / "user_features.parquet")
    logger.info("User features saved", n_users=len(user_features))

    item_features = build_item_features(train)
    save_parquet(item_features, OFFLINE_FEATURES_DIR / "item_features.parquet")
    logger.info("Item features saved", n_items=len(item_features))


if __name__ == "__main__":
    setup_logging()
    build_all_features()
