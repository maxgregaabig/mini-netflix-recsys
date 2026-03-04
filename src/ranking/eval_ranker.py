"""Evaluate the ranker on the test set."""
from __future__ import annotations

import numpy as np

from src.common.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet, save_json
from src.features.feature_defs import RANKING_FEATURE_COLS
from src.ranking.score_ranker import RankerScorer

logger = get_logger(__name__)


def eval_ranker() -> dict:
    """Evaluate the ranker and produce an evaluation report."""
    ensure_dirs()
    df = load_parquet(PROCESSED_DIR / "ranking_dataset.parquet")
    groups = np.load(PROCESSED_DIR / "ranking_groups.npy")

    scorer = RankerScorer()

    X = df[RANKING_FEATURE_COLS].values
    y = df["label"].values
    scores = scorer.score(X)

    # Compute NDCG per group
    ndcg_results: dict[str, list[float]] = {f"ndcg@{k}": [] for k in [5, 10, 20]}

    idx = 0
    for g_size in groups:
        g_labels = y[idx:idx + g_size]
        g_scores = scores[idx:idx + g_size]
        idx += g_size

        # Sort by predicted score
        order = np.argsort(-g_scores)
        sorted_labels = g_labels[order]

        for k in [5, 10, 20]:
            ndcg = _ndcg_at_k(sorted_labels, k)
            ndcg_results[f"ndcg@{k}"].append(ndcg)

    report = {}
    for key, values in ndcg_results.items():
        report[key] = round(float(np.mean(values)), 4)

    save_json(report, REPORTS_DIR / "ranker_eval.json")
    logger.info("Ranker evaluation complete", **report)
    return report


def _ndcg_at_k(sorted_labels: np.ndarray, k: int) -> float:
    """Compute NDCG@K."""
    sorted_labels = sorted_labels[:k]
    gains = 2.0 ** sorted_labels - 1
    discounts = np.log2(np.arange(len(sorted_labels)) + 2)
    dcg = np.sum(gains / discounts)

    ideal = np.sort(sorted_labels)[::-1]
    ideal_gains = 2.0 ** ideal - 1
    ideal_discounts = np.log2(np.arange(len(ideal)) + 2)
    idcg = np.sum(ideal_gains / ideal_discounts)

    return float(dcg / idcg) if idcg > 0 else 0.0


if __name__ == "__main__":
    setup_logging()
    eval_ranker()
