"""Tests for the retrieval pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from src.common.config import FAISS_DIR, MODELS_DIR


def test_embeddings_exist():
    assert (MODELS_DIR / "user_embeddings.npy").exists()
    assert (MODELS_DIR / "item_embeddings.npy").exists()


def test_embedding_shapes():
    user_emb = np.load(MODELS_DIR / "user_embeddings.npy")
    item_emb = np.load(MODELS_DIR / "item_embeddings.npy")
    assert user_emb.ndim == 2
    assert item_emb.ndim == 2
    assert user_emb.shape[1] == item_emb.shape[1], "Embedding dims must match"


def test_faiss_index_exists():
    assert (FAISS_DIR / "index.bin").exists()


def test_retrieval_returns_k_items():
    from src.retrieval.retrieve_candidates import CandidateRetriever

    retriever = CandidateRetriever()
    # Use user_id=1 (should exist in MovieLens small)
    results = retriever.retrieve(user_id=1, top_k=10)
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"


def test_retrieval_returns_tuples():
    from src.retrieval.retrieve_candidates import CandidateRetriever

    retriever = CandidateRetriever()
    results = retriever.retrieve(user_id=1, top_k=5)
    for item_id, score in results:
        assert isinstance(item_id, int)
        assert isinstance(score, float)
