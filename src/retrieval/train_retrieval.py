"""Train the two-tower retrieval model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.config import MODELS_DIR, OFFLINE_FEATURES_DIR, PROCESSED_DIR, ensure_dirs, get_recsys_config
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet, save_json
from src.features.feature_defs import ALL_GENRES, GENRE_TO_IDX
from src.retrieval.model import TwoTowerModel, in_batch_bpr_loss

logger = get_logger(__name__)


class InteractionDataset(Dataset):
    """Dataset of (user_id, user_genre_aff, item_id, item_genre_multihot) tuples."""

    def __init__(
        self,
        events: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        user_id_map: dict[int, int],
        item_id_map: dict[int, int],
    ):
        # Only positive events
        pos = events[events["is_positive"]].copy()
        self.user_ids = pos["user_id"].map(user_id_map).values
        self.item_ids = pos["item_id"].map(item_id_map).values

        # Pre-compute genre vectors
        num_genres = len(ALL_GENRES)
        self.user_genre_aff = np.zeros((len(user_id_map) + 1, num_genres), dtype=np.float32)
        for row in user_features.itertuples():
            uid = user_id_map.get(row.user_id)
            if uid is None:
                continue
            aff = json.loads(row.user_genre_affinity) if isinstance(row.user_genre_affinity, str) else {}
            for g, v in aff.items():
                if g in GENRE_TO_IDX:
                    self.user_genre_aff[uid, GENRE_TO_IDX[g]] = v

        self.item_genre_mh = np.zeros((len(item_id_map) + 1, num_genres), dtype=np.float32)
        for row in item_features.itertuples():
            iid = item_id_map.get(row.item_id)
            if iid is None:
                continue
            mh = json.loads(row.genres_multihot) if isinstance(row.genres_multihot, str) else [0] * num_genres
            self.item_genre_mh[iid] = mh

        # Filter valid
        valid = (~pd.isna(self.user_ids)) & (~pd.isna(self.item_ids))
        self.user_ids = self.user_ids[valid].astype(np.int64)
        self.item_ids = self.item_ids[valid].astype(np.int64)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[int, np.ndarray, int, np.ndarray]:
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        return uid, self.user_genre_aff[uid], iid, self.item_genre_mh[iid]


def train_retrieval() -> None:
    """Train the two-tower model and save it."""
    ensure_dirs()
    cfg = get_recsys_config()["retrieval"]

    train = load_parquet(PROCESSED_DIR / "train.parquet")
    valid = load_parquet(PROCESSED_DIR / "valid.parquet")
    user_features = load_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    item_features = load_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")

    # Build ID maps (0 reserved for padding)
    all_user_ids = sorted(train["user_id"].unique())
    all_item_ids = sorted(item_features["item_id"].unique())
    user_id_map = {uid: i + 1 for i, uid in enumerate(all_user_ids)}
    item_id_map = {iid: i + 1 for i, iid in enumerate(all_item_ids)}

    save_json(
        {str(k): v for k, v in user_id_map.items()},
        MODELS_DIR / "user_ids.json",
    )
    save_json(
        {str(k): v for k, v in item_id_map.items()},
        MODELS_DIR / "item_ids.json",
    )

    num_users = len(user_id_map) + 1
    num_items = len(item_id_map) + 1
    num_genres = len(ALL_GENRES)

    dataset = InteractionDataset(train, user_features, item_features, user_id_map, item_id_map)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    logger.info(
        "Starting retrieval training",
        num_users=num_users,
        num_items=num_items,
        epochs=cfg["num_epochs"],
        device=str(device),
    )

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for user_ids, user_ga, item_ids, item_gm in loader:
            user_ids = user_ids.long().to(device)
            user_ga = user_ga.float().to(device)
            item_ids = item_ids.long().to(device)
            item_gm = item_gm.float().to(device)

            user_emb, item_emb = model(user_ids, user_ga, item_ids, item_gm)
            loss = in_batch_bpr_loss(user_emb, item_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info("Epoch complete", epoch=epoch + 1, avg_loss=round(avg_loss, 5))

    # Save model
    torch.save(model.state_dict(), MODELS_DIR / "two_tower.pt")
    save_json(
        {
            "num_users": num_users,
            "num_items": num_items,
            "num_genres": num_genres,
            "embedding_dim": cfg["embedding_dim"],
            "hidden_dim": cfg["hidden_dim"],
        },
        MODELS_DIR / "two_tower_config.json",
    )
    logger.info("Model saved", path=str(MODELS_DIR / "two_tower.pt"))


if __name__ == "__main__":
    setup_logging()
    train_retrieval()
