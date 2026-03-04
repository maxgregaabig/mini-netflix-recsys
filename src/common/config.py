"""Configuration loading utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
CONFIG_DIR = PROJECT_ROOT / os.getenv("CONFIG_DIR", "configs")


def load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML config file from the configs directory."""
    path = CONFIG_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)


def get_recsys_config() -> dict[str, Any]:
    return load_yaml("recsys.yaml")


def get_serving_config() -> dict[str, Any]:
    return load_yaml("serving.yaml")


def get_features_config() -> dict[str, Any]:
    return load_yaml("features.yaml")


# Derived paths
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OFFLINE_FEATURES_DIR = DATA_DIR / "offline_features"
FAISS_DIR = DATA_DIR / "faiss"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"

ALL_DIRS = [RAW_DIR, PROCESSED_DIR, OFFLINE_FEATURES_DIR, FAISS_DIR, MODELS_DIR, REPORTS_DIR]


def ensure_dirs() -> None:
    """Create all data directories if they don't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
