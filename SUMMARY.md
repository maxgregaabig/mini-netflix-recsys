# Mini-Netflix Recommender System — Project Summary

## What This Is

A complete, production-style movie recommendation system that mimics how Netflix-scale recommenders work, built small enough to run on a laptop. It demonstrates the full ML lifecycle: data ingestion, feature engineering, model training, offline evaluation, real-time serving with performance optimizations, and observability.

---

## The Core Idea: Two-Stage Recommendation

Real-world recommender systems don't score every item for every user. They use a funnel:

```
~10,000 items in catalog
        │
        ▼
  ┌─────────────┐
  │  Retrieval   │  Two-tower neural network + FAISS
  │  (Stage 1)   │  Fast, approximate: "Which 100 items might this user like?"
  └──────┬──────┘
         │ 100 candidates
         ▼
  ┌─────────────┐
  │   Ranking    │  LightGBM with rich features
  │  (Stage 2)   │  Precise, expensive: "What's the best order for these 100?"
  └──────┬──────┘
         │ Top 10-20
         ▼
  ┌─────────────┐
  │  Post-proc   │  Diversity filtering, dedup, row assembly
  └──────┬──────┘
         │
         ▼
   Final response: "Top Picks", "Because You Watched", "Trending"
```

This is exactly how production systems at Netflix, YouTube, and Spotify work.

---

## Component Breakdown

### 1. Data Pipeline (`src/data_prep/`)

**What it does**: Downloads MovieLens ml-latest-small (100K ratings, 610 users, 9,742 movies), converts explicit ratings into implicit feedback events, and splits by time per user.

- `rating >= 4.0` → positive "watch" event
- `rating <= 2.5` → negative "skip" event
- Synthetic watch times generated proportional to rating
- Per-user temporal split ensures no data leakage (train on past, test on future)

**Why it matters**: Real recommenders operate on implicit signals (clicks, watches, skips), not star ratings. This conversion simulates that.

### 2. Feature Store (`src/features/`)

**What it does**: Computes user and item features offline, stores them as Parquet, then materializes to Redis for low-latency online serving.

**User features**:
| Feature | Description |
|---------|-------------|
| `user_watch_count_30d` | Activity level (recency-weighted) |
| `user_genre_affinity` | Normalized genre preference vector |
| `user_mean_rating` | Average rating given |
| `user_recency_mean_days` | How recently they've been active |

**Item features**:
| Feature | Description |
|---------|-------------|
| `item_watch_count` | Popularity signal |
| `genres` / `genres_multihot` | Content metadata |
| `item_mean_rating` | Quality signal |

**Redis layout**: `uf:{user_id}` → JSON, `if:{item_id}` → JSON

**Skew check**: Compares offline training features against online Redis values for a random sample. Outputs `data/reports/skew_report.json`. Training-serving skew is one of the top causes of ML system failures in production.

### 3. Retrieval Model (`src/retrieval/`)

**Architecture**: Two-tower model in PyTorch

```
User Tower:                          Item Tower:
  user_id → Embedding(64)             item_id → Embedding(64)
  genre_affinity → Linear(64)         genre_multihot → Linear(64)
         │                                   │
    Concat(128)                          Concat(128)
         │                                   │
    Linear(128→64) + ReLU               Linear(128→64) + ReLU
    Dropout(0.1)                        Dropout(0.1)
    Linear(64)                          Linear(64)
         │                                   │
    L2 Normalize                        L2 Normalize
         │                                   │
    user_embedding (64d)               item_embedding (64d)
```

**Training objective**: In-batch BPR (Bayesian Personalized Ranking). For each user-item positive pair in a batch, all other items in the batch serve as negatives. This avoids explicit negative sampling and scales well.

**After training**:
- All user and item embeddings computed and saved as `.npy` files
- Item embeddings indexed in FAISS (IndexFlatIP — exact inner product search)
- Retrieval = query FAISS with user embedding → top-100 items in <5ms

### 4. Ranking Model (`src/ranking/`)

**What it does**: Takes the 100 candidates from retrieval and re-orders them using richer features.

**Features fed to LightGBM**:
| Feature | Source | Signal |
|---------|--------|--------|
| `retrieval_score` | FAISS similarity | Embedding-based relevance |
| `item_watch_count` | Feature store | Popularity prior |
| `user_mean_rating` | Feature store | User generosity |
| `item_mean_rating` | Feature store | Item quality |
| `genre_overlap_score` | Computed | User-item genre match |
| `user_recency_mean_days` | Feature store | User activity level |

**Training**: LightGBM with `lambdarank` objective, directly optimizing NDCG@10. Groups defined per-user (each user's candidate list is one group).

### 5. Evaluation (`src/evaluation/`)

**Offline eval** (`offline_eval.py`):
- Retrieval: Recall@10, @20, @50 — "Did the retriever find the relevant items?"
- Ranking: NDCG@5, @10, @20 — "Did the ranker put relevant items at the top?"
- Coverage: Fraction of catalog recommended across all users
- Genre diversity: Entropy of genre distribution in recommendations

**A/B simulation** (`ab_simulation.py`):
- Randomly assigns test users to control (retrieval-only ordering) vs treatment (retrieval + ranking)
- Computes NDCG@10 lift with 1000-iteration bootstrap confidence intervals
- Reports whether the lift is statistically significant

### 6. Real-Time Serving (`src/serving/`)

This is the most production-relevant part. The FastAPI server implements several patterns used in real ML serving systems:

#### Request Flow
```
Request → Middleware (logging, backpressure) → Route handler
    → Check response cache (30s TTL)
    → Fetch user features from Redis (fallback: offline parquet)
    → Get/compute user embedding (cached in Redis, 10min TTL)
    → FAISS retrieval → 100 candidates
    → Build ranking features for each candidate
    → Submit to dynamic batcher → LightGBM batch predict
    → Post-process (diversity filter, build rows)
    → Cache response → Return JSON
```

#### Dynamic Batching (`batching.py`)
Under concurrent load, individual ranking requests are collected for a short window (`max_wait_ms=10`), merged into a single LightGBM predict call, and results dispatched back to each request via asyncio futures. This amortizes Python/model overhead.

#### Backpressure (`middleware.py`)
- Concurrency semaphore: limits simultaneous requests
- Queue depth limit: ranking batcher rejects when full
- Per-request timeout: 2 seconds
- Returns HTTP 429 with JSON error when overloaded

#### Caching (`cache.py`)
Three cache layers, each with Prometheus hit/miss tracking:
- User features: avoids Redis round-trip (10min TTL)
- User embeddings: avoids recomputation (10min TTL)
- Full response: absorbs repeated requests (30s TTL)

#### Observability
- Structured JSON logging with `request_id` and per-stage timings
- Prometheus `/metrics` endpoint with histograms for latency, queue depth, batch sizes, cache rates

### 7. Demo UI (`src/demo/streamlit_app.py`)

Streamlit app that calls the API and displays three recommendation rows (Top Picks, Because You Watched, Trending) with movie titles, scores, and genres. Includes an event logging form.

### 8. Load Testing (`src/loadtest/`)

Two options:
- `python_load_test.py`: Standalone async script (200 requests, 20 concurrent), reports p50/p95/p99
- `locustfile.py`: Full Locust load test with configurable users and duration

---

## File Tree (60 files)

```
mini_netflix_recsys/
├── .env.example              # Environment variable defaults
├── .gitignore                # Ignores data/, __pycache__, etc.
├── Dockerfile                # Python 3.11 slim image
├── Makefile                  # All pipeline commands
├── README.md                 # Full documentation with architecture diagrams
├── SUMMARY.md                # This file
├── docker-compose.yml        # API + Redis + optional trainer/demo
├── pyproject.toml            # Dependencies and tool config
│
├── configs/
│   ├── features.yaml         # Feature definitions, Redis key prefixes
│   ├── recsys.yaml           # Training hyperparameters, data config
│   └── serving.yaml          # Server, batching, cache, backpressure config
│
├── data/                     # All generated artifacts (gitignored)
│   ├── raw/                  # MovieLens download
│   ├── processed/            # events.parquet, train/valid/test splits
│   ├── offline_features/     # user_features.parquet, item_features.parquet
│   ├── faiss/                # index.bin
│   ├── models/               # two_tower.pt, lgbm_ranker.txt, embeddings
│   └── reports/              # offline_eval.json, ab_report.json, etc.
│
├── src/
│   ├── common/
│   │   ├── config.py         # YAML loading, path constants
│   │   ├── logging.py        # structlog JSON logging setup
│   │   ├── metrics.py        # Prometheus counter/histogram/gauge definitions
│   │   ├── schemas.py        # Pydantic models for API
│   │   └── utils.py          # Parquet/JSON I/O, cosine similarity
│   │
│   ├── data_prep/
│   │   ├── download_movielens.py   # Downloads + extracts ml-latest-small
│   │   ├── prepare_events.py       # Ratings → implicit events + metadata merge
│   │   └── split_train_valid_test.py # Per-user temporal split
│   │
│   ├── features/
│   │   ├── feature_defs.py         # Genre constants, feature column lists
│   │   ├── build_offline_features.py # Computes user/item features → Parquet
│   │   ├── materialize_to_redis.py  # Writes features to Redis
│   │   └── skew_check.py           # Compares offline vs online values
│   │
│   ├── retrieval/
│   │   ├── model.py                # TwoTowerModel, UserTower, ItemTower, BPR loss
│   │   ├── train_retrieval.py      # Training loop with DataLoader
│   │   ├── embed_users_items.py    # Compute all embeddings → .npy
│   │   ├── build_faiss_index.py    # Build IndexFlatIP from item embeddings
│   │   └── retrieve_candidates.py  # CandidateRetriever class (FAISS queries)
│   │
│   ├── ranking/
│   │   ├── build_ranking_dataset.py # Retrieval candidates + features + labels
│   │   ├── train_ranker.py         # LightGBM lambdarank training
│   │   ├── score_ranker.py         # RankerScorer class (batch predict)
│   │   └── eval_ranker.py          # NDCG evaluation per group
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # recall@k, ndcg@k, coverage, diversity
│   │   ├── offline_eval.py         # Full retrieval+ranking evaluation
│   │   └── ab_simulation.py        # Control vs treatment with bootstrap CIs
│   │
│   ├── serving/
│   │   ├── app.py                  # FastAPI app factory + lifespan
│   │   ├── routes.py               # /health, /recs/home, /log_event, /metrics
│   │   ├── dependencies.py         # Singleton model registry, cache, batcher
│   │   ├── cache.py                # CacheClient: Redis + local + TTL + metrics
│   │   ├── batching.py             # DynamicBatcher: async queue → batch predict
│   │   ├── models.py               # ModelRegistry: loads retriever + ranker
│   │   ├── feature_client.py       # FeatureClient: Redis + offline fallback
│   │   └── middleware.py           # Observability + backpressure middleware
│   │
│   ├── demo/
│   │   └── streamlit_app.py        # Demo UI with rec rows + event logging
│   │
│   └── loadtest/
│       ├── locustfile.py           # Locust load test definition
│       └── python_load_test.py     # Standalone async load test + report
│
└── tests/
    ├── test_data_prep.py           # Validates event columns, splits
    ├── test_features.py            # Validates feature schemas, JSON parsing
    ├── test_retrieval.py           # Validates embeddings, FAISS, top-K results
    ├── test_ranking.py             # Validates model loading, scoring shapes
    └── test_serving_smoke.py       # FastAPI TestClient: health, recs, events
```

---

## How to Run (Copy-Paste Commands)

### Local Setup (Windows or Linux)

```bash
cd mini_netflix_recsys

# Install everything
pip install -e ".[dev,demo,loadtest]"

# Start Redis (needs Docker)
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Run the full pipeline end-to-end
make train
# This runs: download → prepare events → split → build features →
#            materialize to Redis → train two-tower → compute embeddings →
#            build FAISS index → build ranking dataset → train LightGBM →
#            offline evaluation → A/B simulation

# Start the API server
make serve
# → http://localhost:8000/health
# → http://localhost:8000/recs/home?user_id=1
# → http://localhost:8000/metrics

# Start the demo UI (separate terminal)
make demo
# → http://localhost:8501

# Run load test (separate terminal, while API is running)
make loadtest

# Run tests (after make train has completed)
make test
```

### Docker Compose

```bash
# Train (one-off)
docker compose --profile train run trainer

# Serve
docker compose up --build
# → API at http://localhost:8000
```

---

## Key Technologies

| Component | Technology | Why |
|-----------|-----------|-----|
| Retrieval model | PyTorch | Flexible neural architecture for two-tower |
| ANN search | FAISS (faiss-cpu) | Industry standard for embedding similarity search |
| Ranking model | LightGBM | Fast, accurate gradient boosting; native lambdarank support |
| Feature store (online) | Redis | Sub-millisecond key-value lookups |
| Feature store (offline) | Parquet | Columnar format, efficient for analytics |
| API server | FastAPI | Async Python, auto-docs, Pydantic validation |
| Metrics | prometheus-client | Standard format for monitoring stacks |
| Logging | structlog | Structured JSON logs with context |
| Demo | Streamlit | Quick interactive UI from Python |
| Load test | Locust + httpx | Industry-standard + lightweight async option |
| Container | Docker + Compose | Reproducible local environment |

---

## What Makes This Production-Style

1. **Two-stage architecture** — not a single model, but a retrieval→ranking funnel like real systems
2. **Feature store pattern** — offline compute, online serve, skew detection
3. **Dynamic batching** — real serving optimization used at scale
4. **Backpressure** — graceful degradation under load instead of cascading failures
5. **Multi-layer caching** — with TTLs and hit-rate monitoring
6. **Prometheus observability** — per-stage latency histograms, not just request-level
7. **Offline evaluation** — proper metrics (NDCG, Recall) + A/B simulation with statistical rigor
8. **Structured logging** — request tracing with IDs and timing breakdowns
9. **Configuration-driven** — YAML configs, not hardcoded values
10. **Idempotent scripts** — safe to re-run any step

---

## Stats

- **~3,200 lines** of Python across 35 source files
- **5 test files** covering data, features, retrieval, ranking, and serving
- **3 YAML configs** for features, training, and serving
- **60 files total** including Docker, Makefile, configs, and docs
- Trains in **~2-5 minutes** on a laptop CPU
- Serves recommendations in **~20-80ms** per request
