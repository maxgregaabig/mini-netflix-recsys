"""Train a LightGBM LambdaRank ranker."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from src.common.config import MODELS_DIR, PROCESSED_DIR, ensure_dirs, get_recsys_config
from src.common.logging import get_logger, setup_logging
from src.common.utils import load_parquet
from src.features.feature_defs import RANKING_FEATURE_COLS

logger = get_logger(__name__)


def train_ranker() -> None:
    """Train LightGBM ranker with lambdarank objective."""
    ensure_dirs()
    cfg = get_recsys_config()["ranking"]["lgbm_params"]

    df = load_parquet(PROCESSED_DIR / "ranking_dataset.parquet")
    groups = np.load(PROCESSED_DIR / "ranking_groups.npy")

    X = df[RANKING_FEATURE_COLS].values
    y = df["label"].values

    # Split groups for train/valid (80/20 of groups)
    n_groups = len(groups)
    split_idx = int(n_groups * 0.8)

    train_end = int(groups[:split_idx].sum())
    X_train, y_train = X[:train_end], y[:train_end]
    X_valid, y_valid = X[train_end:], y[train_end:]
    groups_train = groups[:split_idx]
    groups_valid = groups[split_idx:]

    logger.info(
        "Training LightGBM ranker",
        train_samples=len(X_train),
        valid_samples=len(X_valid),
        train_groups=len(groups_train),
        valid_groups=len(groups_valid),
        features=RANKING_FEATURE_COLS,
    )

    train_data = lgb.Dataset(
        X_train, label=y_train, group=groups_train,
        feature_name=RANKING_FEATURE_COLS,
    )
    valid_data = lgb.Dataset(
        X_valid, label=y_valid, group=groups_valid,
        feature_name=RANKING_FEATURE_COLS,
        reference=train_data,
    )

    params = {
        "objective": cfg["objective"],
        "metric": cfg["metric"],
        "eval_at": cfg["eval_at"],
        "num_leaves": cfg["num_leaves"],
        "learning_rate": cfg["learning_rate"],
        "min_child_samples": cfg["min_child_samples"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "verbose": cfg["verbose"],
    }

    callbacks = [lgb.log_evaluation(period=50)]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=cfg["n_estimators"],
        valid_sets=[valid_data],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    model_path = str(MODELS_DIR / "lgbm_ranker.txt")
    model.save_model(model_path)
    logger.info("Ranker saved", path=model_path, best_iter=model.best_iteration)


if __name__ == "__main__":
    setup_logging()
    train_ranker()
