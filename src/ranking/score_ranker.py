"""Score candidates with the trained LightGBM ranker."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from src.common.config import MODELS_DIR
from src.common.logging import get_logger
from src.features.feature_defs import RANKING_FEATURE_COLS

logger = get_logger(__name__)


class RankerScorer:
    """Wraps a LightGBM ranker model for batch scoring."""

    def __init__(self) -> None:
        self.model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_ranker.txt"))
        logger.info("RankerScorer initialized")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score a batch of feature vectors.

        Args:
            features: shape (N, len(RANKING_FEATURE_COLS))

        Returns:
            scores: shape (N,)
        """
        return self.model.predict(features)

    def score_candidates(
        self,
        candidates: list[dict],
    ) -> list[tuple[int, float]]:
        """Score a list of candidate dicts and return sorted (item_id, score) pairs.

        Each candidate dict must have keys matching RANKING_FEATURE_COLS plus 'item_id'.
        """
        if not candidates:
            return []

        features = np.array(
            [[c[f] for f in RANKING_FEATURE_COLS] for c in candidates],
            dtype=np.float64,
        )
        scores = self.score(features)

        results = [(c["item_id"], float(s)) for c, s in zip(candidates, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
