"""Caching layer with Redis + in-memory fallback and hit-rate tracking."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import redis

from src.common.logging import get_logger
from src.common.metrics import CACHE_HITS, CACHE_MISSES

logger = get_logger(__name__)


class CacheClient:
    """Multi-layer cache: user features, user embeddings, and full response cache."""

    def __init__(self, config: dict) -> None:
        self.config = config
        redis_cfg = config.get("redis", {})
        self._redis = redis.Redis(
            host=redis_cfg.get("host", os.getenv("REDIS_HOST", "localhost")),
            port=redis_cfg.get("port", int(os.getenv("REDIS_PORT", "6379"))),
            db=redis_cfg.get("db", int(os.getenv("REDIS_DB", "0"))),
            decode_responses=True,
            socket_connect_timeout=2,
        )
        self._local_cache: dict[str, tuple[float, Any]] = {}
        cache_cfg = config.get("cache", {})
        self.uf_ttl = cache_cfg.get("user_features_ttl", 600)
        self.ue_ttl = cache_cfg.get("user_embedding_ttl", 600)
        self.resp_ttl = cache_cfg.get("response_ttl", 30)

    def _local_get(self, key: str) -> Any | None:
        entry = self._local_cache.get(key)
        if entry is None:
            return None
        expire_at, val = entry
        if time.time() > expire_at:
            del self._local_cache[key]
            return None
        return val

    def _local_set(self, key: str, value: Any, ttl: int) -> None:
        self._local_cache[key] = (time.time() + ttl, value)

    # --- User features ---
    def get_user_features(self, user_id: int) -> dict | None:
        cache_key = f"uf:{user_id}"
        # L1: local
        val = self._local_get(cache_key)
        if val is not None:
            CACHE_HITS.labels(cache_type="user_features").inc()
            return val
        # L2: Redis
        try:
            raw = self._redis.get(cache_key)
            if raw is not None:
                parsed = json.loads(raw)
                self._local_set(cache_key, parsed, self.uf_ttl)
                CACHE_HITS.labels(cache_type="user_features").inc()
                return parsed
        except redis.RedisError:
            pass
        CACHE_MISSES.labels(cache_type="user_features").inc()
        return None

    # --- User embedding ---
    def get_user_embedding(self, user_id: int) -> list[float] | None:
        cache_key = f"ue:{user_id}"
        val = self._local_get(cache_key)
        if val is not None:
            CACHE_HITS.labels(cache_type="user_embedding").inc()
            return val
        try:
            raw = self._redis.get(cache_key)
            if raw is not None:
                parsed = json.loads(raw)
                self._local_set(cache_key, parsed, self.ue_ttl)
                CACHE_HITS.labels(cache_type="user_embedding").inc()
                return parsed
        except redis.RedisError:
            pass
        CACHE_MISSES.labels(cache_type="user_embedding").inc()
        return None

    def set_user_embedding(self, user_id: int, embedding: list[float]) -> None:
        cache_key = f"ue:{user_id}"
        self._local_set(cache_key, embedding, self.ue_ttl)
        try:
            self._redis.setex(cache_key, self.ue_ttl, json.dumps(embedding))
        except redis.RedisError:
            pass

    # --- Response cache ---
    def get_response(self, user_id: int, seed_item_id: int | None = None) -> dict | None:
        h = hashlib.md5(f"{user_id}:{seed_item_id}".encode()).hexdigest()[:12]
        cache_key = f"resp:{h}"
        val = self._local_get(cache_key)
        if val is not None:
            CACHE_HITS.labels(cache_type="response").inc()
            return val
        CACHE_MISSES.labels(cache_type="response").inc()
        return None

    def set_response(self, user_id: int, response: dict, seed_item_id: int | None = None) -> None:
        h = hashlib.md5(f"{user_id}:{seed_item_id}".encode()).hexdigest()[:12]
        cache_key = f"resp:{h}"
        self._local_set(cache_key, response, self.resp_ttl)

    # --- Item features (bulk, from Redis) ---
    def get_item_features(self, item_id: int) -> dict | None:
        cache_key = f"if:{item_id}"
        val = self._local_get(cache_key)
        if val is not None:
            return val
        try:
            raw = self._redis.get(cache_key)
            if raw is not None:
                parsed = json.loads(raw)
                self._local_set(cache_key, parsed, self.uf_ttl)
                return parsed
        except redis.RedisError:
            pass
        return None
