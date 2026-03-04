"""FastAPI route handlers."""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest

from src.common.config import PROCESSED_DIR
from src.common.logging import get_logger
from src.common.metrics import RANKING_LATENCY, RETRIEVAL_LATENCY
from src.common.schemas import (
    ErrorResponse,
    EventPayload,
    EventResponse,
    HealthResponse,
    HomeRecsResponse,
    MovieRec,
    RecRow,
)
from src.serving.dependencies import get_batcher, get_cache, get_features, get_models

logger = get_logger(__name__)
router = APIRouter()


_trending_cache: tuple[float, list[tuple[dict, float]]] | None = None
_TRENDING_TTL = 60  # refresh every 60 seconds


def _get_global_trending(
    features: "FeatureClient",
    exclude_ids: set[int] | None = None,
    n: int = 10,
) -> list[tuple[dict, float]]:
    """Return the top-N most popular items globally, cached for 60s."""
    global _trending_cache
    now = time.monotonic()

    if _trending_cache is None or (now - _trending_cache[0]) > _TRENDING_TTL:
        # Build from offline item features
        all_items: list[tuple[dict, int]] = []
        for item_id, ifeat in features._offline_items.items():
            genres_str = ifeat.get("genres", "")
            all_items.append((
                {
                    "item_id": item_id,
                    "title": ifeat.get("title", f"Movie {item_id}"),
                    "genres": [g.strip() for g in genres_str.split("|") if g.strip()] if isinstance(genres_str, str) else [],
                    "popularity": ifeat.get("item_watch_count", 0),
                },
                ifeat.get("item_watch_count", 0),
            ))
        all_items.sort(key=lambda x: x[1], reverse=True)
        # Keep top 50 so we have room to filter watched + diversify
        top_items = [(meta, float(pop)) for meta, pop in all_items[:50]]
        _trending_cache = (now, top_items)

    items = _trending_cache[1]

    # Filter out watched
    if exclude_ids:
        items = [(m, s) for m, s in items if m["item_id"] not in exclude_ids]

    # Diversify
    genre_counts: dict[str, int] = {}
    result = []
    for meta, score in items:
        g = meta["genres"][0] if meta["genres"] else ""
        if genre_counts.get(g, 0) >= 3 and g:
            continue
        genre_counts[g] = genre_counts.get(g, 0) + 1
        result.append((meta, score))
        if len(result) >= n:
            break
    return result


def _get_user_live_history(user_id: int) -> tuple[set[int], int | None]:
    """Read the live event log and return (watched_item_ids, last_watched_item_id) for a user."""
    log_path = PROCESSED_DIR / "events_live.log"
    watched: list[int] = []
    if not log_path.exists():
        return set(), None
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("user_id") == user_id and ev.get("event_type") == "watch":
                    watched.append(ev["item_id"])
    except Exception:
        return set(), None
    last = watched[-1] if watched else None
    return set(watched), last


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode(), media_type="text/plain")


@router.get("/recs/home", response_model=HomeRecsResponse)
async def home_recs(
    request: Request,
    user_id: int = Query(..., description="User ID"),
    seed_item_id: int | None = Query(None, description="Seed item for 'Because You Watched'"),
) -> HomeRecsResponse | JSONResponse:
    """Generate home page recommendations with multiple rows."""
    request_id = getattr(request.state, "request_id", "unknown")
    start = time.monotonic()
    timings: dict[str, float] = {}

    models = get_models()
    cache = get_cache()
    features = get_features()
    batcher = get_batcher()

    # Read live event history for this user
    watched_ids, last_watched = _get_user_live_history(user_id)
    # Auto-pick seed from last watched if not provided
    if seed_item_id is None and last_watched is not None:
        seed_item_id = last_watched

    # Check response cache
    cached = cache.get_response(user_id, seed_item_id)
    if cached is not None:
        cached["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return HomeRecsResponse(**cached)

    # 1. Fetch user features
    t0 = time.monotonic()
    user_feat = features.get_user_features(user_id)
    timings["feature_fetch_ms"] = round((time.monotonic() - t0) * 1000, 1)

    if user_feat is None:
        user_feat = {
            "user_mean_rating": 3.5,
            "user_recency_mean_days": 30.0,
            "user_genre_affinity": "{}",
        }

    user_genre_aff = user_feat.get("user_genre_affinity", "{}")
    if isinstance(user_genre_aff, str):
        user_genre_aff = json.loads(user_genre_aff)

    # 2. Retrieve candidates
    t0 = time.monotonic()
    candidates = models.retriever.retrieve(user_id, top_k=100)
    retrieval_time = time.monotonic() - t0
    timings["retrieval_ms"] = round(retrieval_time * 1000, 1)
    RETRIEVAL_LATENCY.observe(retrieval_time)

    if not candidates:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(error="no_candidates", detail="No candidates found").model_dump(),
        )

    # 3. Build ranking features
    ranking_inputs = []
    candidate_meta: list[dict] = []
    for item_id, ret_score in candidates:
        ifeat = features.get_item_features(item_id)
        if ifeat is None:
            continue

        genres_str = ifeat.get("genres", "")
        overlap = sum(
            user_genre_aff.get(g.strip(), 0.0)
            for g in genres_str.split("|") if g.strip()
        ) if isinstance(genres_str, str) else 0.0

        ranking_inputs.append([
            ret_score,
            ifeat.get("item_watch_count", 0),
            user_feat.get("user_mean_rating", 3.5),
            ifeat.get("item_mean_rating", 3.0),
            overlap,
            user_feat.get("user_recency_mean_days", 30.0),
        ])
        candidate_meta.append({
            "item_id": item_id,
            "title": ifeat.get("title", f"Movie {item_id}"),
            "genres": [g.strip() for g in genres_str.split("|") if g.strip()] if isinstance(genres_str, str) else [],
            "popularity": ifeat.get("item_watch_count", 0),
        })

    if not ranking_inputs:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(error="no_features", detail="No item features found").model_dump(),
        )

    # 4. Score via dynamic batcher
    t0 = time.monotonic()
    feature_matrix = np.array(ranking_inputs, dtype=np.float64)

    try:
        scores = await batcher.submit(
            request_id=request_id,
            features=feature_matrix,
            timeout=2.0,
        )
    except asyncio.QueueFull:
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(error="queue_full", detail="Ranking queue is full").model_dump(),
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content=ErrorResponse(error="timeout", detail="Ranking timed out").model_dump(),
        )

    ranking_time = time.monotonic() - t0
    timings["ranking_ms"] = round(ranking_time * 1000, 1)
    RANKING_LATENCY.observe(ranking_time)

    # 5. Post-processing
    t0 = time.monotonic()

    scored = list(zip(candidate_meta, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    # Filter out already-watched items from live log
    if watched_ids:
        scored = [(m, s) for m, s in scored if m["item_id"] not in watched_ids]

    # Diversity: limit same genre repetitions
    def diversify(items: list[tuple[dict, float]], max_per_genre: int = 3, n: int = 10) -> list[tuple[dict, float]]:
        genre_counts: dict[str, int] = defaultdict(int)
        result = []
        for meta, score in items:
            dominated_genre = meta["genres"][0] if meta["genres"] else ""
            if genre_counts[dominated_genre] >= max_per_genre and dominated_genre:
                continue
            genre_counts[dominated_genre] += 1
            result.append((meta, score))
            if len(result) >= n:
                break
        return result

    # Top Picks
    top_picks = diversify(scored, n=10)

    # Because You Watched: use seed_item or last-watched heuristic
    byw_candidates = []
    if seed_item_id is not None:
        similar = models.retriever.retrieve_similar_items(seed_item_id, top_k=20)
        for sim_id, sim_score in similar:
            meta_match = next((m for m, _ in scored if m["item_id"] == sim_id), None)
            if meta_match:
                byw_candidates.append((meta_match, sim_score))
            else:
                ifeat = features.get_item_features(sim_id)
                if ifeat:
                    genres_str = ifeat.get("genres", "")
                    byw_candidates.append(({
                        "item_id": sim_id,
                        "title": ifeat.get("title", f"Movie {sim_id}"),
                        "genres": [g.strip() for g in genres_str.split("|") if g.strip()] if isinstance(genres_str, str) else [],
                        "popularity": ifeat.get("item_watch_count", 0),
                    }, sim_score))

    if not byw_candidates:
        # Fallback: take items ranked 10-20
        byw_candidates = scored[10:30]
    byw = diversify(byw_candidates, n=10)

    # Trending: global popularity (not per-user candidates)
    trending = _get_global_trending(features, watched_ids)

    timings["postprocess_ms"] = round((time.monotonic() - t0) * 1000, 1)

    # Build response — sigmoid normalization to 0-100 "match %"
    # Calibrated against empirical score distribution (median ~ -3.7, range ~ -10 to +4)
    # sigmoid centered at median, scaled so top items land 70-98%, weak items 20-50%
    import math

    def _score_to_match_pct(raw: float) -> float:
        # Shift and scale: center=median, scale controls steepness
        center = -3.7
        scale = 2.5
        return 100.0 / (1.0 + math.exp(-(raw - center) / scale))

    def to_recs(items: list[tuple[dict, float]]) -> list[MovieRec]:
        if not items:
            return []
        return [
            MovieRec(
                item_id=meta["item_id"],
                title=meta.get("title", ""),
                score=round(_score_to_match_pct(float(s)), 1),
                genres=meta.get("genres", []),
            )
            for meta, s in items
        ]

    def to_trending_recs(items: list[tuple[dict, float]]) -> list[MovieRec]:
        """Trending uses watch count as score, not ranker output."""
        if not items:
            return []
        return [
            MovieRec(
                item_id=meta["item_id"],
                title=meta.get("title", ""),
                score=round(float(s)),
                genres=meta.get("genres", []),
            )
            for meta, s in items
        ]

    total_ms = round((time.monotonic() - start) * 1000, 1)

    response_data = {
        "user_id": user_id,
        "rows": [
            RecRow(row_name="top_picks", items=to_recs(top_picks)),
            RecRow(row_name="because_you_watched", items=to_recs(byw)),
            RecRow(row_name="trending", items=to_trending_recs(trending)),
        ],
        "latency_ms": total_ms,
    }

    # Cache the response
    cache.set_response(user_id, {
        "user_id": user_id,
        "rows": [r.model_dump() for r in response_data["rows"]],
        "latency_ms": 0,
    }, seed_item_id)

    logger.info(
        "recs_served",
        request_id=request_id,
        user_id=user_id,
        **timings,
        total_ms=total_ms,
    )

    return HomeRecsResponse(**response_data)


@router.post("/log_event", response_model=EventResponse)
async def log_event(payload: EventPayload) -> EventResponse:
    """Log an event to a local append-only file (simulates event stream)."""
    log_path = PROCESSED_DIR / "events_live.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload.model_dump(), default=str) + "\n"
    with open(log_path, "a") as f:
        f.write(line)
    return EventResponse()
