"""End-to-end offline evaluation: retrieval + ranking pipeline."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from src.common.config import OFFLINE_FEATURES_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, get_recsys_config
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet, save_json
from src.evaluation.metrics import coverage, diversity_entropy, ndcg_at_k, recall_at_k
from src.ranking.score_ranker import RankerScorer
from src.retrieval.retrieve_candidates import CandidateRetriever

logger = get_logger(__name__)


def run_offline_eval() -> dict:
    """Run full offline evaluation on test set."""
    ensure_dirs()
    cfg = get_recsys_config()

    test = load_parquet(PROCESSED_DIR / "test.parquet")
    user_features = load_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    item_features = load_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")

    retriever = CandidateRetriever()
    ranker = RankerScorer()

    # Build feature lookups
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

    # Ground truth: positive test items per user
    test_positives = test[test["is_positive"]].groupby("user_id")["item_id"].apply(set).to_dict()

    recall_k = cfg["evaluation"]["recall_at_k"]
    ndcg_k = cfg["evaluation"]["ndcg_at_k"]
    top_k = cfg["retrieval"]["top_k"]

    retrieval_recalls: dict[int, list[float]] = {k: [] for k in recall_k}
    ranking_ndcgs: dict[int, list[float]] = {k: [] for k in ndcg_k}
    all_recommended: set[int] = set()
    genre_counts: dict[str, int] = defaultdict(int)

    n_eval = 0
    for user_id, relevant_items in test_positives.items():
        if user_id not in user_feat_dict:
            continue

        # Retrieval
        candidates = retriever.retrieve(user_id, top_k=top_k)
        retrieved_ids = [c[0] for c in candidates]

        for k in recall_k:
            retrieval_recalls[k].append(recall_at_k(retrieved_ids, relevant_items, k))

        # Ranking
        user_feat = user_feat_dict[user_id]
        ranking_candidates = []
        for item_id, ret_score in candidates:
            ifeat = item_feat_dict.get(item_id, {})
            if not ifeat:
                continue

            def _genre_overlap(aff: dict, genres: str) -> float:
                if not aff or not genres:
                    return 0.0
                return sum(aff.get(g.strip(), 0.0) for g in genres.split("|"))

            ranking_candidates.append({
                "item_id": item_id,
                "retrieval_score": ret_score,
                "item_watch_count": ifeat.get("item_watch_count", 0),
                "user_mean_rating": user_feat["user_mean_rating"],
                "item_mean_rating": ifeat.get("item_mean_rating", 0.0),
                "genre_overlap_score": _genre_overlap(user_feat["user_genre_affinity"], ifeat.get("genres", "")),
                "user_recency_mean_days": user_feat["user_recency_mean_days"],
            })

        ranked = ranker.score_candidates(ranking_candidates)
        ranked_ids = [r[0] for r in ranked]

        for k in ndcg_k:
            ranking_ndcgs[k].append(ndcg_at_k(ranked_ids, relevant_items, k))

        all_recommended.update(ranked_ids[:20])
        for item_id in ranked_ids[:20]:
            genres = item_feat_dict.get(item_id, {}).get("genres", "")
            for g in genres.split("|"):
                g = g.strip()
                if g:
                    genre_counts[g] += 1

        n_eval += 1

    report = {
        "n_users_evaluated": n_eval,
        "retrieval": {f"recall@{k}": round(float(np.mean(v)), 4) for k, v in retrieval_recalls.items()},
        "ranking": {f"ndcg@{k}": round(float(np.mean(v)), 4) for k, v in ranking_ndcgs.items()},
        "coverage": round(coverage(all_recommended, set(item_feat_dict.keys())), 4),
        "genre_diversity_entropy": round(diversity_entropy(genre_counts), 4),
    }

    save_json(report, REPORTS_DIR / "offline_eval.json")
    logger.info("Offline evaluation complete", **report)
    return report


if __name__ == "__main__":
    setup_logging()
    run_offline_eval()
