"""Training-serving skew check: compare offline vs online feature values."""
from __future__ import annotations

import json

import numpy as np

from src.common.config import OFFLINE_FEATURES_DIR, REPORTS_DIR, ensure_dirs
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet, save_json
from src.features.materialize_to_redis import get_redis_client

logger = get_logger(__name__)

SAMPLE_SIZE = 50


def check_skew() -> dict:
    """Compare offline feature values with online (Redis) values for a sample of users/items."""
    ensure_dirs()
    r = get_redis_client()
    report: dict = {"user_skew": [], "item_skew": [], "summary": {}}

    # User features
    user_df = load_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    sample_users = user_df.sample(min(SAMPLE_SIZE, len(user_df)), random_state=42)

    user_mismatches = 0
    for row in sample_users.itertuples():
        online_raw = r.get(f"uf:{row.user_id}")
        if online_raw is None:
            user_mismatches += 1
            report["user_skew"].append({
                "user_id": int(row.user_id),
                "issue": "missing_from_redis",
            })
            continue

        online = json.loads(online_raw)
        offline_mean = float(row.user_mean_rating) if row.user_mean_rating == row.user_mean_rating else 0.0
        online_mean = online.get("user_mean_rating", 0.0)

        if abs(offline_mean - online_mean) > 0.01:
            user_mismatches += 1
            report["user_skew"].append({
                "user_id": int(row.user_id),
                "feature": "user_mean_rating",
                "offline": offline_mean,
                "online": online_mean,
                "diff": round(abs(offline_mean - online_mean), 4),
            })

    # Item features
    item_df = load_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")
    sample_items = item_df.sample(min(SAMPLE_SIZE, len(item_df)), random_state=42)

    item_mismatches = 0
    for row in sample_items.itertuples():
        online_raw = r.get(f"if:{row.item_id}")
        if online_raw is None:
            item_mismatches += 1
            report["item_skew"].append({
                "item_id": int(row.item_id),
                "issue": "missing_from_redis",
            })
            continue

        online = json.loads(online_raw)
        offline_pop = int(row.item_watch_count)
        online_pop = online.get("item_watch_count", 0)

        if offline_pop != online_pop:
            item_mismatches += 1
            report["item_skew"].append({
                "item_id": int(row.item_id),
                "feature": "item_watch_count",
                "offline": offline_pop,
                "online": online_pop,
            })

    report["summary"] = {
        "users_sampled": len(sample_users),
        "user_mismatches": user_mismatches,
        "items_sampled": len(sample_items),
        "item_mismatches": item_mismatches,
        "skew_detected": user_mismatches > 0 or item_mismatches > 0,
    }

    save_json(report, REPORTS_DIR / "skew_report.json")
    logger.info("Skew check complete", **report["summary"])
    return report


if __name__ == "__main__":
    setup_logging()
    check_skew()
