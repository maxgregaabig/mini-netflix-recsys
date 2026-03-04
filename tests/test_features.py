"""Tests for offline feature building."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.common.config import OFFLINE_FEATURES_DIR


def test_user_features_exist():
    path = OFFLINE_FEATURES_DIR / "user_features.parquet"
    assert path.exists(), f"Expected {path} to exist. Run 'make features' first."


def test_user_features_schema():
    df = pd.read_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    assert len(df) > 0, "User features is empty"
    required = {"user_id", "user_watch_count_30d", "user_genre_affinity", "user_mean_rating"}
    assert required.issubset(set(df.columns))


def test_user_genre_affinity_is_valid_json():
    df = pd.read_parquet(OFFLINE_FEATURES_DIR / "user_features.parquet")
    sample = df["user_genre_affinity"].dropna().head(10)
    for val in sample:
        parsed = json.loads(val)
        assert isinstance(parsed, dict)


def test_item_features_exist():
    path = OFFLINE_FEATURES_DIR / "item_features.parquet"
    assert path.exists(), f"Expected {path} to exist. Run 'make features' first."


def test_item_features_schema():
    df = pd.read_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")
    assert len(df) > 0, "Item features is empty"
    required = {"item_id", "item_watch_count", "genres", "genres_multihot", "item_mean_rating"}
    assert required.issubset(set(df.columns))


def test_item_multihot_is_valid():
    df = pd.read_parquet(OFFLINE_FEATURES_DIR / "item_features.parquet")
    sample = df["genres_multihot"].dropna().head(10)
    for val in sample:
        parsed = json.loads(val)
        assert isinstance(parsed, list)
        assert all(v in (0, 1) for v in parsed)
