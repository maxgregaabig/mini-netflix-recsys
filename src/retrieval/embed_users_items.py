"""Compute and save user/item embeddings from the trained two-tower model."""
from __future__ import annotations

import json

import numpy as np
import torch

from src.common.config import MODELS_DIR, OFFLINE_FEATURES_DIR, ensure_dirs
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_json, load_parquet
from src.features.feature_defs import ALL_GENRES, GENRE_TO_IDX
from src.retrieval.model import TwoTowerModel

logger = get_logger(__name__)


def compute_embeddings() -> None:
    """Compute all user and item embeddings and save as .npy files."""
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model config and weights
    model_cfg = load_json(MODELS_DIR / "two_tower_config.json")
    model = TwoTowerModel(**model_cfg).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "two_tower.pt", map_location=device, weights_only=True))
    model.eval()

    user_id_map = {int(k): v for k, v in load_json(MODELS_DIR / "user_ids.json").items()}
    item_id_map = {int(k): v for k, v in load_json(MODELS_DIR / "item_ids.json").items()}
    num_genres = len(ALL_GENRES)

    # Load features
    user_features = load_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    item_features = load_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")

    # User embeddings
    num_users = model_cfg["num_users"]
    user_genre_aff = np.zeros((num_users, num_genres), dtype=np.float32)
    for row in user_features.itertuples():
        uid = user_id_map.get(row.user_id)
        if uid is None:
            continue
        aff = json.loads(row.user_genre_affinity) if isinstance(row.user_genre_affinity, str) else {}
        for g, v in aff.items():
            if g in GENRE_TO_IDX:
                user_genre_aff[uid, GENRE_TO_IDX[g]] = v

    logger.info("Computing user embeddings", n=num_users)
    with torch.no_grad():
        user_ids_t = torch.arange(num_users).long().to(device)
        user_ga_t = torch.tensor(user_genre_aff).float().to(device)
        # Process in chunks to avoid OOM
        chunk = 512
        user_embs = []
        for i in range(0, num_users, chunk):
            emb = model.user_tower(user_ids_t[i:i+chunk], user_ga_t[i:i+chunk])
            user_embs.append(emb.cpu().numpy())
        user_embeddings = np.concatenate(user_embs, axis=0)

    # Item embeddings
    num_items = model_cfg["num_items"]
    item_genre_mh = np.zeros((num_items, num_genres), dtype=np.float32)
    for row in item_features.itertuples():
        iid = item_id_map.get(row.item_id)
        if iid is None:
            continue
        mh = json.loads(row.genres_multihot) if isinstance(row.genres_multihot, str) else [0] * num_genres
        item_genre_mh[iid] = mh

    logger.info("Computing item embeddings", n=num_items)
    with torch.no_grad():
        item_ids_t = torch.arange(num_items).long().to(device)
        item_gm_t = torch.tensor(item_genre_mh).float().to(device)
        item_embs = []
        for i in range(0, num_items, chunk):
            emb = model.item_tower(item_ids_t[i:i+chunk], item_gm_t[i:i+chunk])
            item_embs.append(emb.cpu().numpy())
        item_embeddings = np.concatenate(item_embs, axis=0)

    np.save(MODELS_DIR / "user_embeddings.npy", user_embeddings)
    np.save(MODELS_DIR / "item_embeddings.npy", item_embeddings)

    logger.info(
        "Embeddings saved",
        user_shape=user_embeddings.shape,
        item_shape=item_embeddings.shape,
    )


if __name__ == "__main__":
    setup_logging()
    compute_embeddings()
