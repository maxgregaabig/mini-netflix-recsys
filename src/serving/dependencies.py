"""FastAPI dependency injection: shared model registry, cache, batcher."""
from __future__ import annotations

from src.serving.batching import DynamicBatcher
from src.serving.cache import CacheClient
from src.serving.feature_client import FeatureClient
from src.serving.models import ModelRegistry

# Singletons initialized at startup
_model_registry: ModelRegistry | None = None
_cache_client: CacheClient | None = None
_feature_client: FeatureClient | None = None
_batcher: DynamicBatcher | None = None


def init_dependencies(config: dict) -> None:
    """Initialize all serving dependencies."""
    global _model_registry, _cache_client, _feature_client, _batcher

    _model_registry = ModelRegistry()
    _cache_client = CacheClient(config)
    _feature_client = FeatureClient(_cache_client)

    batching_cfg = config.get("batching", {})
    bp_cfg = config.get("backpressure", {})
    _batcher = DynamicBatcher(
        score_fn=_model_registry.ranker.score,
        max_wait_ms=batching_cfg.get("max_wait_ms", 10),
        max_batch_size=batching_cfg.get("max_batch_size", 64),
        max_queue_depth=bp_cfg.get("max_queue_depth", 100),
    )


def get_models() -> ModelRegistry:
    assert _model_registry is not None
    return _model_registry


def get_cache() -> CacheClient:
    assert _cache_client is not None
    return _cache_client


def get_features() -> FeatureClient:
    assert _feature_client is not None
    return _feature_client


def get_batcher() -> DynamicBatcher:
    assert _batcher is not None
    return _batcher
