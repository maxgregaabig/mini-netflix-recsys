"""Build the ranking dataset: for each user, create labeled candidate lists."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import MODELS_DIR, OFFLINE_FEATURES_DIR, PROCESSED_DIR, ensure_dirs, get_recsys_config
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_json, load_parquet, save_parquet
from src.features.feature_defs import ALL_GENRES, GENRE_TO_IDX
from src.retrieval.retrieve_candidates import CandidateRetriever

logger = get_logger(__name__)


def compute_genre_overlap(user_genre_aff: dict, item_genres: str) -> float:
    """Compute overlap between user genre affinity and item genres."""
    if not user_genre_aff or not isinstance(item_genres, str):
        return 0.0
    item_gs = [g.strip() for g in item_genres.split("|")]
    score = sum(user_genre_aff.get(g, 0.0) for g in item_gs)
    return round(score, 4)


def build_ranking_dataset() -> None:
    """Build ranking training data with retrieval candidates + features."""
    ensure_dirs()
    cfg = get_recsys_config()["ranking"]

    train = load_parquet(PROCESSED_DIR / "train.parquet")
    user_features = load_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    item_features = load_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")

    retriever = CandidateRetriever()

    # Build lookup dicts
    user_feat_dict = {}
    for row in user_features.itertuples():
        aff = json.loads(row.user_genre_affinity) if isinstance(row.user_genre_affinity, str) else {}
        user_feat_dict[row.user_id] = {
            "user_mean_rating": float(row.user_mean_rating) if row.user_mean_rating == row.user_mean_rating else 0.0,
            "user_recency_mean_days": float(row.user_recency_mean_days) if row.user_recency_mean_days == row.user_recency_mean_days else 0.0,
            "user_genre_affinity": aff,
        }

    item_feat_dict = {}
    for row in item_features.itertuples():
        item_feat_dict[row.item_id] = {
            "item_watch_count": int(row.item_watch_count),
            "item_mean_rating": float(row.item_mean_rating) if row.item_mean_rating == row.item_mean_rating else 0.0,
            "genres": row.genres if isinstance(row.genres, str) else "",
        }

    # Positive items per user
    positives = train[train["is_positive"]].groupby("user_id")["item_id"].apply(set).to_dict()

    rows = []
    group_sizes = []
    users_processed = 0

    for user_id, pos_items in positives.items():
        if user_id not in user_feat_dict:
            continue

        candidates = retriever.retrieve(user_id, top_k=cfg["num_candidates"])
        if not candidates:
            continue

        user_feat = user_feat_dict[user_id]
        user_group = []

        for item_id, retrieval_score in candidates:
            item_feat = item_feat_dict.get(item_id, {})
            if not item_feat:
                continue

            label = 1 if item_id in pos_items else 0
            genre_overlap = compute_genre_overlap(
                user_feat["user_genre_affinity"],
                item_feat.get("genres", ""),
            )

            user_group.append({
                "user_id": user_id,
                "item_id": item_id,
                "label": label,
                "retrieval_score": retrieval_score,
                "item_watch_count": item_feat.get("item_watch_count", 0),
                "user_mean_rating": user_feat["user_mean_rating"],
                "item_mean_rating": item_feat.get("item_mean_rating", 0.0),
                "genre_overlap_score": genre_overlap,
                "user_recency_mean_days": user_feat["user_recency_mean_days"],
            })

        if user_group:
            rows.extend(user_group)
            group_sizes.append(len(user_group))
            users_processed += 1

    df = pd.DataFrame(rows)
    save_parquet(df, PROCESSED_DIR / "ranking_dataset.parquet")

    group_arr = np.array(group_sizes, dtype=np.int32)
    np.save(PROCESSED_DIR / "ranking_groups.npy", group_arr)

    logger.info(
        "Ranking dataset built",
        total_rows=len(df),
        n_users=users_processed,
        n_groups=len(group_sizes),
        positives=int(df["label"].sum()),
        negatives=int((df["label"] == 0).sum()),
    )


if __name__ == "__main__":
    setup_logging()
    build_ranking_dataset()
