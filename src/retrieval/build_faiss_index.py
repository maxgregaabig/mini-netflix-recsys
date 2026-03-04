"""Build a FAISS index from item embeddings for fast ANN retrieval."""
from __future__ import annotations

import faiss
import numpy as np

from src.common.config import FAISS_DIR, MODELS_DIR, ensure_dirs
from src.common.logging import get_logger, setup_logging

logger = get_logger(__name__)


def build_index() -> None:
    """Build and save a FAISS IndexFlatIP (inner product) from item embeddings."""
    ensure_dirs()

    item_embeddings = np.load(MODELS_DIR / "item_embeddings.npy")
    logger.info("Building FAISS index", shape=item_embeddings.shape)

    # Embeddings are L2-normalized, so inner product = cosine similarity
    dim = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_embeddings.astype(np.float32))

    faiss.write_index(index, str(FAISS_DIR / "index.bin"))
    logger.info("FAISS index saved", n_vectors=index.ntotal, dim=dim)


if __name__ == "__main__":
    setup_logging()
    build_index()
