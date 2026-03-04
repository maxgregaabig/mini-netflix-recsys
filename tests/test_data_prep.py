"""Tests for data preparation pipeline."""
from __future__ import annotations

import pandas as pd
import pytest

from src.common.config import PROCESSED_DIR


def test_events_parquet_exists():
    path = PROCESSED_DIR / "events.parquet"
    assert path.exists(), f"Expected {path} to exist. Run 'make data' first."


def test_events_has_correct_columns():
    df = pd.read_parquet(PROCESSED_DIR / "events.parquet")
    required = {"user_id", "item_id", "timestamp", "event_type", "watch_time", "rating"}
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"


def test_events_has_positive_and_negative():
    df = pd.read_parquet(PROCESSED_DIR / "events.parquet")
    assert df["is_positive"].any(), "No positive events found"
    assert df["is_negative"].any(), "No negative events found"


def test_splits_exist():
    for split in ["train", "valid", "test"]:
        path = PROCESSED_DIR / f"{split}.parquet"
        assert path.exists(), f"Expected {path} to exist"


def test_splits_have_data():
    for split in ["train", "valid", "test"]:
        df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")
        assert len(df) > 0, f"{split} split is empty"


def test_train_is_largest():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    valid = pd.read_parquet(PROCESSED_DIR / "valid.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    assert len(train) > len(valid), "Train should be larger than valid"
    assert len(train) > len(test), "Train should be larger than test"
