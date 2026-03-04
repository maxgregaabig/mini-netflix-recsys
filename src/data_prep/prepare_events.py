"""Convert MovieLens ratings into implicit feedback events."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_DIR, RAW_DIR, ensure_dirs, get_recsys_config
from src.common.logging import get_logger, setup_logging
from src.common.utils import save_parquet

logger = get_logger(__name__)


def prepare_events() -> pd.DataFrame:
    """Convert ratings.csv to events with implicit feedback signals.

    - rating >= 4.0 → positive "watch" event
    - rating <= 2.5 → negative event (kept for training signal but flagged)
    - 2.5 < rating < 4.0 → weak positive, kept as "watch" with lower synthetic watch_time
    """
    ensure_dirs()
    cfg = get_recsys_config()["data"]

    ratings_path = RAW_DIR / "ml-latest-small" / "ratings.csv"
    movies_path = RAW_DIR / "ml-latest-small" / "movies.csv"

    logger.info("Loading ratings", path=str(ratings_path))
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Create events
    events = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="s")

    # Event type based on rating
    pos_thresh = cfg["positive_threshold"]
    neg_thresh = cfg["negative_threshold"]

    events["event_type"] = "watch"
    events.loc[events["rating"] <= neg_thresh, "event_type"] = "skip"

    # Synthetic watch_time: higher rating → longer watch
    rng = np.random.default_rng(42)
    base_time = events["rating"] / 5.0 * 120  # up to 120 minutes
    noise = rng.normal(0, 10, size=len(events))
    events["watch_time"] = np.clip(base_time + noise, 1, 200).round(1)

    # Filter: keep positives and negatives, drop the middle for cleaner signal
    events["is_positive"] = events["rating"] >= pos_thresh
    events["is_negative"] = events["rating"] <= neg_thresh

    # Merge movie metadata
    movies = movies.rename(columns={"movieId": "item_id", "title": "title"})
    events = events.merge(movies[["item_id", "title", "genres"]], on="item_id", how="left")

    # Sort by timestamp
    events = events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Save
    save_parquet(events, PROCESSED_DIR / "events.parquet")
    movies.to_parquet(PROCESSED_DIR / "movies.parquet", index=False)

    logger.info(
        "Events prepared",
        total=len(events),
        positive=int(events["is_positive"].sum()),
        negative=int(events["is_negative"].sum()),
        users=events["user_id"].nunique(),
        items=events["item_id"].nunique(),
    )
    return events


if __name__ == "__main__":
    setup_logging()
    prepare_events()
