"""Tests for the LightGBM ranking pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.common.config import MODELS_DIR


def test_ranker_model_exists():
    assert (MODELS_DIR / "lgbm_ranker.txt").exists()


def test_ranker_can_score():
    from src.ranking.score_ranker import RankerScorer

    scorer = RankerScorer()
    # 6 features: retrieval_score, item_watch_count, user_mean_rating,
    #             item_mean_rating, genre_overlap_score, user_recency_mean_days
    features = np.array([
        [0.8, 100, 3.5, 4.0, 0.6, 10.0],
        [0.5, 50, 3.0, 3.5, 0.3, 20.0],
        [0.9, 200, 4.0, 4.5, 0.9, 5.0],
    ])
    scores = scorer.score(features)
    assert scores.shape == (3,), f"Expected shape (3,), got {scores.shape}"
    assert all(isinstance(s, (float, np.floating)) for s in scores)


def test_ranker_score_candidates():
    from src.ranking.score_ranker import RankerScorer

    scorer = RankerScorer()
    candidates = [
        {
            "item_id": 1,
            "retrieval_score": 0.8,
            "item_watch_count": 100,
            "user_mean_rating": 3.5,
            "item_mean_rating": 4.0,
            "genre_overlap_score": 0.6,
            "user_recency_mean_days": 10.0,
        },
        {
            "item_id": 2,
            "retrieval_score": 0.5,
            "item_watch_count": 50,
            "user_mean_rating": 3.0,
            "item_mean_rating": 3.5,
            "genre_overlap_score": 0.3,
            "user_recency_mean_days": 20.0,
        },
    ]
    results = scorer.score_candidates(candidates)
    assert len(results) == 2
    # Results should be sorted by score descending
    assert results[0][1] >= results[1][1]
