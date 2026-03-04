.PHONY: setup data features train_retrieval train_ranker eval train serve demo loadtest test clean

# ─── Setup ───────────────────────────────────────────────────────
setup:
	pip install -e ".[dev,demo,loadtest]"

# ─── Data ────────────────────────────────────────────────────────
data:
	python -m src.data_prep.download_movielens
	python -m src.data_prep.prepare_events
	python -m src.data_prep.split_train_valid_test

# ─── Features ────────────────────────────────────────────────────
features:
	python -m src.features.build_offline_features
	python -m src.features.materialize_to_redis

skew_check:
	python -m src.features.skew_check

# ─── Retrieval ───────────────────────────────────────────────────
train_retrieval:
	python -m src.retrieval.train_retrieval
	python -m src.retrieval.embed_users_items
	python -m src.retrieval.build_faiss_index

# ─── Ranking ─────────────────────────────────────────────────────
train_ranker:
	python -m src.ranking.build_ranking_dataset
	python -m src.ranking.train_ranker

# ─── Evaluation ──────────────────────────────────────────────────
eval:
	python -m src.ranking.eval_ranker
	python -m src.evaluation.offline_eval
	python -m src.evaluation.ab_simulation

# ─── Full Training Pipeline ─────────────────────────────────────
train: data features train_retrieval train_ranker eval
	@echo "Full training pipeline complete!"
	@echo "Reports saved to data/reports/"

# ─── Serving ─────────────────────────────────────────────────────
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

serve-docker:
	docker compose up --build redis api

# ─── Demo ────────────────────────────────────────────────────────
demo:
	streamlit run src/demo/streamlit_app.py --server.port 8501

# ─── Load Test ───────────────────────────────────────────────────
loadtest:
	python -m src.loadtest.python_load_test

loadtest-locust:
	locust -f src/loadtest/locustfile.py --headless -u 50 -r 10 --run-time 30s --host http://localhost:8000

# ─── Tests ───────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-smoke:
	pytest tests/test_serving_smoke.py -v

# ─── Clean ───────────────────────────────────────────────────────
clean:
	rm -rf data/raw data/processed data/offline_features data/faiss data/models data/reports
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
