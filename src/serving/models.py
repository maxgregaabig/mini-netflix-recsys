"""Model loading: FAISS index, embeddings, LightGBM ranker, two-tower model."""
from __future__ import annotations

from src.common.logging import get_logger
from src.ranking.score_ranker import RankerScorer
from src.retrieval.retrieve_candidates import CandidateRetriever

logger = get_logger(__name__)


class ModelRegistry:
    """Holds all loaded models and indices for serving."""

    def __init__(self) -> None:
        logger.info("Loading models...")
        self.retriever = CandidateRetriever()
        self.ranker = RankerScorer()
        logger.info("All models loaded")
