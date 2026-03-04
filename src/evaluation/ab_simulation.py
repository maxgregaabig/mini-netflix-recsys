"""A/B test simulation: compare retrieval-only (control) vs retrieval+ranking (treatment)."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from src.common.config import OFFLINE_FEATURES_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, get_recsys_config
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet, save_json
from src.evaluation.metrics import ndcg_at_k
from src.ranking.score_ranker import RankerScorer
from src.retrieval.retrieve_candidates import CandidateRetriever

logger = get_logger(__name__)


def run_ab_simulation() -> dict:
    """Simulate A/B test between control (retrieval-only) and treatment (retrieval+ranking)."""
    ensure_dirs()
    cfg = get_recsys_config()
    n_users = cfg["evaluation"]["ab_num_users"]
    n_bootstrap = cfg["evaluation"]["ab_bootstrap_iterations"]

    test = load_parquet(PROCESSED_DIR / "test.parquet")
    user_features = load_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    item_features = load_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")

    retriever = CandidateRetriever()
    ranker = RankerScorer()

    # Feature lookups
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

    test_positives = test[test["is_positive"]].groupby("user_id")["item_id"].apply(set).to_dict()

    # Random user assignment
    rng = np.random.default_rng(42)
    all_users = [u for u in test_positives if u in user_feat_dict]
    sampled = rng.choice(all_users, size=min(n_users, len(all_users)), replace=False)
    assignment = rng.choice(["control", "treatment"], size=len(sampled))

    control_ndcg_10: list[float] = []
    treatment_ndcg_10: list[float] = []

    for user_id, group in zip(sampled, assignment):
        relevant = test_positives[user_id]
        candidates = retriever.retrieve(user_id, top_k=100)

        if group == "control":
            # Control: retrieval order only
            ranked_ids = [c[0] for c in candidates]
            ndcg = ndcg_at_k(ranked_ids, relevant, 10)
            control_ndcg_10.append(ndcg)
        else:
            # Treatment: retrieval + ranking
            user_feat = user_feat_dict[user_id]
            ranking_candidates = []
            for item_id, ret_score in candidates:
                ifeat = item_feat_dict.get(item_id, {})
                if not ifeat:
                    continue
                overlap = sum(
                    user_feat["user_genre_affinity"].get(g.strip(), 0.0)
                    for g in ifeat.get("genres", "").split("|")
                    if g.strip()
                )
                ranking_candidates.append({
                    "item_id": item_id,
                    "retrieval_score": ret_score,
                    "item_watch_count": ifeat.get("item_watch_count", 0),
                    "user_mean_rating": user_feat["user_mean_rating"],
                    "item_mean_rating": ifeat.get("item_mean_rating", 0.0),
                    "genre_overlap_score": overlap,
                    "user_recency_mean_days": user_feat["user_recency_mean_days"],
                })

            ranked = ranker.score_candidates(ranking_candidates)
            ranked_ids = [r[0] for r in ranked]
            ndcg = ndcg_at_k(ranked_ids, relevant, 10)
            treatment_ndcg_10.append(ndcg)

    # Bootstrap confidence intervals
    control_arr = np.array(control_ndcg_10)
    treatment_arr = np.array(treatment_ndcg_10)

    control_mean = float(np.mean(control_arr))
    treatment_mean = float(np.mean(treatment_arr))
    lift = (treatment_mean - control_mean) / max(control_mean, 1e-8)

    # Bootstrap for CI on lift
    lifts = []
    for _ in range(n_bootstrap):
        c_sample = rng.choice(control_arr, size=len(control_arr), replace=True)
        t_sample = rng.choice(treatment_arr, size=len(treatment_arr), replace=True)
        c_m = np.mean(c_sample)
        t_m = np.mean(t_sample)
        lifts.append((t_m - c_m) / max(c_m, 1e-8))

    lifts = np.array(lifts)
    ci_lower = float(np.percentile(lifts, 2.5))
    ci_upper = float(np.percentile(lifts, 97.5))

    report = {
        "control": {
            "n_users": len(control_ndcg_10),
            "mean_ndcg_10": round(control_mean, 4),
        },
        "treatment": {
            "n_users": len(treatment_ndcg_10),
            "mean_ndcg_10": round(treatment_mean, 4),
        },
        "lift": round(lift, 4),
        "bootstrap_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "significant": ci_lower > 0,
    }

    save_json(report, REPORTS_DIR / "ab_report.json")
    logger.info("A/B simulation complete", **report)
    return report


if __name__ == "__main__":
    setup_logging()
    run_ab_simulation()
