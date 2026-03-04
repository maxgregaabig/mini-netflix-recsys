"""Prometheus metrics definitions."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "requests_total",
    "Total request count",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Current ranking queue depth",
)

BATCH_SIZE = Histogram(
    "batch_size",
    "Ranking batch sizes",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Cache hit count",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Cache miss count",
    ["cache_type"],
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "FAISS retrieval latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

RANKING_LATENCY = Histogram(
    "ranking_latency_seconds",
    "LightGBM ranking latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

FEATURE_FETCH_LATENCY = Histogram(
    "feature_fetch_latency_seconds",
    "Feature fetch from Redis latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05),
)
