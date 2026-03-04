# Mini-Netflix Recommender System

A production-style recommendation system with **two-stage retrieval + ranking**, a **mini feature store**, **real-time inference with dynamic batching**, and a **Streamlit demo UI**.

Built with PyTorch, LightGBM, FAISS, FastAPI, and Redis on MovieLens data.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Request                          │
│                    GET /recs/home?user_id=42                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Server                              │
│  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐  │
│  │Middleware │→ │  Backpres- │→ │  Request   │→ │  Response   │  │
│  │(logging,  │  │  sure      │  │  Router    │  │  Cache      │  │
│  │ tracing)  │  │  (429)     │  │            │  │  (30s TTL)  │  │
│  └──────────┘  └────────────┘  └─────┬──────┘  └────────────┘  │
└──────────────────────────────────────┬──────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│  Feature Store   │  │   Stage 1: Retrieval  │  │   Stage 2: Rank  │
│                  │  │                       │  │                  │
│  Redis (online)  │  │  Two-Tower Model      │  │  LightGBM        │
│  ┌────────────┐  │  │  (PyTorch)            │  │  (LambdaRank)    │
│  │uf:{user_id}│  │  │         │             │  │       │          │
│  │if:{item_id}│  │  │         ▼             │  │       ▼          │
│  │ue:{user_id}│  │  │  User Embedding       │  │  Dynamic Batcher │
│  └────────────┘  │  │         │             │  │  (batch scoring) │
│                  │  │         ▼             │  │       │          │
│  Parquet         │  │  FAISS ANN Index      │  │       ▼          │
│  (offline        │  │  → Top-100 candidates │  │  Re-ranked Top-N │
│   fallback)      │  │                       │  │                  │
└──────────────────┘  └──────────────────────┘  └──────────────────┘
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │  Post-Processing │
                                              │  - Dedup          │
                                              │  - Diversity      │
                                              │  - Build rows     │
                                              └──────────────────┘
```

## How It Works

### Two-Stage Recommendation

**Stage 1 — Retrieval (Two-Tower Model)**
- A user tower and item tower each produce 64-dim L2-normalized embeddings
- User tower: `user_id embedding + genre affinity → MLP → embedding`
- Item tower: `item_id embedding + genre multi-hot → MLP → embedding`
- Trained with **in-batch BPR loss** (Bayesian Personalized Ranking)
- At serving time, user embedding queries a **FAISS IndexFlatIP** for top-100 candidates in <5ms

**Stage 2 — Ranking (LightGBM LambdaRank)**
- Candidates from retrieval are scored by a LightGBM ranker
- Features: retrieval similarity, item popularity, user/item mean ratings, genre overlap, recency
- Objective: `lambdarank` optimizing NDCG
- Batch scoring via **dynamic batcher** for throughput efficiency

### Feature Store

```
Offline Pipeline:                    Online Serving:

events.parquet                      Redis
    │                                 │
    ▼                                 ▼
build_offline_features.py    ──→   uf:{user_id}  (user features JSON)
    │                              if:{item_id}  (item features JSON)
    ▼                              ue:{user_id}  (user embedding, cached)
user_features.parquet
item_features.parquet         Fallback: offline parquet loaded in memory
```

- **Offline**: Features computed from training data, stored as Parquet
- **Online**: Materialized to Redis via `materialize_to_redis.py`
- **Skew check**: `skew_check.py` compares offline vs online values for a sample
- **Fallback**: If Redis is unavailable, falls back to in-memory offline features

### Dynamic Batching

```
Request 1 ─┐
Request 2 ──┼─→ [Collect for max_wait_ms] ─→ Merge features ─→ LightGBM.predict() ─→ Dispatch
Request 3 ──┘                                   (single call)
```

- Collects ranking requests over a configurable window (`max_wait_ms=10`)
- Merges feature matrices into one batch call to LightGBM
- Dispatches per-request scores via asyncio futures
- Reduces per-request overhead under concurrent load

### Backpressure & Caching

- **Queue depth limit**: Returns HTTP 429 when ranking queue is full
- **Concurrency limit**: Semaphore-based middleware rejects excess requests
- **Per-request timeout**: 2s default
- **Cache layers**: User features (10min TTL), user embeddings (10min), full response (30s)
- **Cache hit rate** tracked in Prometheus metrics

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (or Docker)
- ~500MB disk space for data + models

### Option 1: Local (Recommended for Development)

```bash
# 1. Install dependencies
make setup

# 2. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 3. Run the full pipeline (download data → train → evaluate)
make train

# 4. Start the API server
make serve

# 5. (In another terminal) Start the demo UI
make demo

# 6. Open http://localhost:8501 in your browser
```

### Option 2: Docker Compose

```bash
# Run the training pipeline first
docker compose --profile train run trainer

# Start the API + Redis
docker compose up --build

# API at http://localhost:8000
# Demo at http://localhost:8501 (if using demo profile)
```

### Step-by-Step Make Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install all Python dependencies |
| `make data` | Download MovieLens + prepare events + split |
| `make features` | Build offline features + materialize to Redis |
| `make train_retrieval` | Train two-tower model + embeddings + FAISS index |
| `make train_ranker` | Build ranking dataset + train LightGBM |
| `make eval` | Run offline evaluation + A/B simulation |
| `make train` | Full end-to-end pipeline (all of the above) |
| `make serve` | Start FastAPI server on port 8000 |
| `make demo` | Start Streamlit UI on port 8501 |
| `make loadtest` | Run async load test (200 requests) |
| `make test` | Run pytest suite |
| `make skew_check` | Compare offline vs online feature values |
| `make clean` | Remove all generated data |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics (text format) |
| `/recs/home?user_id=1` | GET | Home page recommendations |
| `/recs/home?user_id=1&seed_item_id=50` | GET | With "Because You Watched" seed |
| `/log_event` | POST | Log a user event |

### Example Response

```json
{
  "user_id": 1,
  "rows": [
    {
      "row_name": "top_picks",
      "items": [
        {"item_id": 318, "title": "Shawshank Redemption", "score": 0.9234, "genres": ["Crime", "Drama"]},
        ...
      ]
    },
    {"row_name": "because_you_watched", "items": [...]},
    {"row_name": "trending", "items": [...]}
  ],
  "latency_ms": 45.2
}
```

## Reports & Metrics

After running `make train`, reports are saved to `data/reports/`:

| File | Contents |
|------|----------|
| `offline_eval.json` | Recall@K (retrieval), NDCG@K (ranking), coverage, diversity |
| `ab_report.json` | A/B simulation: control vs treatment lift + bootstrap CI |
| `ranker_eval.json` | LightGBM NDCG on validation groups |
| `skew_report.json` | Training-serving skew check results |
| `loadtest_report.json` | Latency percentiles + throughput |

## Prometheus Metrics

Available at `GET /metrics`:

- `requests_total` — Total requests by method/endpoint/status
- `request_latency_seconds` — End-to-end latency histogram
- `queue_depth` — Current ranking batch queue depth
- `batch_size` — Histogram of batch sizes processed
- `cache_hits_total` / `cache_misses_total` — By cache type (user_features, user_embedding, response)
- `retrieval_latency_seconds` — FAISS retrieval time
- `ranking_latency_seconds` — LightGBM scoring time
- `feature_fetch_latency_seconds` — Redis feature fetch time

## Load Testing

```bash
# Quick async test (200 requests, 20 concurrent)
make loadtest

# Full Locust test (50 users, 30 seconds)
make loadtest-locust
```

Results include p50/p95/p99 latency, throughput (RPS), and error rate.

## Project Structure

```
mini_netflix_recsys/
├── README.md
├── Makefile
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── configs/
│   ├── recsys.yaml          # Training & model hyperparameters
│   ├── serving.yaml         # API server, batching, cache config
│   └── features.yaml        # Feature definitions & Redis keys
├── data/                    # All generated data (gitignored)
│   ├── raw/                 # MovieLens download
│   ├── processed/           # Events, splits, ranking dataset
│   ├── offline_features/    # User/item feature Parquet files
│   ├── faiss/               # FAISS index
│   ├── models/              # Trained models, embeddings, ID maps
│   └── reports/             # Evaluation & load test reports
├── src/
│   ├── common/              # Shared utilities
│   │   ├── config.py        # YAML config loading, paths
│   │   ├── logging.py       # Structured JSON logging
│   │   ├── schemas.py       # Pydantic request/response models
│   │   ├── utils.py         # Parquet/JSON I/O helpers
│   │   └── metrics.py       # Prometheus metric definitions
│   ├── data_prep/           # Data ingestion pipeline
│   │   ├── download_movielens.py
│   │   ├── prepare_events.py
│   │   └── split_train_valid_test.py
│   ├── features/            # Mini feature store
│   │   ├── feature_defs.py  # Feature names, genre constants
│   │   ├── build_offline_features.py
│   │   ├── materialize_to_redis.py
│   │   └── skew_check.py
│   ├── retrieval/           # Two-tower model + FAISS
│   │   ├── model.py         # PyTorch two-tower architecture
│   │   ├── train_retrieval.py
│   │   ├── embed_users_items.py
│   │   ├── build_faiss_index.py
│   │   └── retrieve_candidates.py
│   ├── ranking/             # LightGBM ranker
│   │   ├── build_ranking_dataset.py
│   │   ├── train_ranker.py
│   │   ├── score_ranker.py
│   │   └── eval_ranker.py
│   ├── evaluation/          # Offline metrics & A/B simulation
│   │   ├── metrics.py
│   │   ├── offline_eval.py
│   │   └── ab_simulation.py
│   ├── serving/             # Real-time inference server
│   │   ├── app.py           # FastAPI app + lifespan
│   │   ├── routes.py        # API endpoints
│   │   ├── dependencies.py  # Dependency injection
│   │   ├── cache.py         # Multi-layer caching
│   │   ├── batching.py      # Dynamic batch scoring
│   │   ├── models.py        # Model loading
│   │   ├── feature_client.py# Online feature fetching
│   │   └── middleware.py    # Observability + backpressure
│   ├── demo/
│   │   └── streamlit_app.py # Demo UI
│   └── loadtest/
│       ├── locustfile.py    # Locust load test
│       └── python_load_test.py  # Standalone async load test
└── tests/
    ├── test_data_prep.py
    ├── test_features.py
    ├── test_retrieval.py
    ├── test_ranking.py
    └── test_serving_smoke.py
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| BPR with in-batch negatives | Simple, effective for implicit feedback; avoids explicit negative sampling |
| FAISS IndexFlatIP | Exact search is fast enough for ~10K items; no need for approximate methods |
| LambdaRank objective | Directly optimizes NDCG, the standard ranking metric |
| Async dynamic batching | Amortizes LightGBM predict overhead under concurrent load |
| Redis + Parquet fallback | Redis for low-latency serving; Parquet as cold fallback avoids hard Redis dependency |
| Response cache (30s TTL) | Absorbs repeated requests for the same user in quick succession |
| Prometheus metrics | Industry standard; compatible with Grafana dashboards |

## What I'd Do Next in Production

- **Event streaming**: Replace append-only log with Kafka/Kinesis for real-time event ingestion
- **Online learning**: Incremental model updates from streaming events (e.g., hourly retraining)
- **Model registry**: MLflow or similar for versioning models, A/B model deployment
- **Drift monitoring**: Track feature distribution shifts, model prediction drift over time
- **Approximate NN**: Switch to FAISS IVF or ScaNN for larger item catalogs (>1M items)
- **GPU inference**: Serve the user tower on GPU for real-time embedding computation
- **Feature platform**: Feast or Tecton for managed feature pipelines with point-in-time joins
- **Kubernetes**: Horizontal scaling with HPA based on queue depth / latency metrics
- **Shadow mode**: Run new models in shadow alongside production for comparison
- **Explore/exploit**: Thompson sampling or epsilon-greedy for balancing recommendations
- **User sessions**: Session-based recommendations using transformer sequence models
- **Cold start**: Content-based fallback for new users/items with no interaction history
