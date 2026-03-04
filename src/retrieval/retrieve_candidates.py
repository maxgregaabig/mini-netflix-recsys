"""Retrieve top-K candidate items for a user via FAISS."""
from __future__ import annotations

import faiss
import numpy as np

from src.common.config import FAISS_DIR, MODELS_DIR
from src.common.logging import get_logger
from src.common.utils import load_json

logger = get_logger(__name__)


class CandidateRetriever:
    """FAISS-backed candidate retriever."""

    def __init__(self) -> None:
        self.index = faiss.read_index(str(FAISS_DIR / "index.bin"))
        self.user_embeddings = np.load(MODELS_DIR / "user_embeddings.npy")
        self.item_embeddings = np.load(MODELS_DIR / "item_embeddings.npy")
        self.user_id_map = {int(k): v for k, v in load_json(MODELS_DIR / "user_ids.json").items()}
        self.item_id_map = {int(k): v for k, v in load_json(MODELS_DIR / "item_ids.json").items()}
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        logger.info(
            "CandidateRetriever initialized",
            n_users=len(self.user_id_map),
            n_items=self.index.ntotal,
        )

    def get_user_embedding(self, user_id: int) -> np.ndarray | None:
        """Get precomputed user embedding."""
        idx = self.user_id_map.get(user_id)
        if idx is None:
            return None
        return self.user_embeddings[idx]

    def retrieve(self, user_id: int, top_k: int = 100) -> list[tuple[int, float]]:
        """Retrieve top-K items for a user.

        Returns list of (item_id, similarity_score) tuples.
        """
        user_emb = self.get_user_embedding(user_id)
        if user_emb is None:
            logger.warning("Unknown user, returning popular items", user_id=user_id)
            return self._popular_fallback(top_k)

        user_emb = user_emb.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(user_emb, top_k + 1)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx <= 0:  # skip padding
                continue
            item_id = self.reverse_item_map.get(idx)
            if item_id is not None:
                results.append((item_id, float(score)))

        return results[:top_k]

    def retrieve_similar_items(self, item_id: int, top_k: int = 20) -> list[tuple[int, float]]:
        """Retrieve items similar to a given item."""
        idx = self.item_id_map.get(item_id)
        if idx is None:
            return []

        item_emb = self.item_embeddings[idx].reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(item_emb, top_k + 1)

        results = []
        for score, i in zip(scores[0], indices[0]):
            if i <= 0 or i == idx:
                continue
            real_id = self.reverse_item_map.get(i)
            if real_id is not None:
                results.append((real_id, float(score)))

        return results[:top_k]

    def _popular_fallback(self, top_k: int) -> list[tuple[int, float]]:
        """Return random items as fallback for unknown users."""
        all_items = list(self.reverse_item_map.values())
        selected = all_items[:top_k]
        return [(iid, 0.5) for iid in selected]
