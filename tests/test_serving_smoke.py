"""Smoke tests for the serving API using FastAPI TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_metrics(client: TestClient):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "requests_total" in resp.text


def test_home_recs(client: TestClient):
    resp = client.get("/recs/home?user_id=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == 1
    assert "rows" in data
    row_names = [r["row_name"] for r in data["rows"]]
    assert "top_picks" in row_names
    assert "trending" in row_names


def test_home_recs_with_seed(client: TestClient):
    resp = client.get("/recs/home?user_id=1&seed_item_id=1")
    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data


def test_log_event(client: TestClient):
    resp = client.post("/log_event", json={
        "user_id": 1,
        "item_id": 1,
        "event_type": "watch",
        "watch_time": 90.0,
        "rating": 4.5,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "logged"


def test_unknown_user_returns_recs(client: TestClient):
    resp = client.get("/recs/home?user_id=999999")
    # Should still return something (fallback behavior)
    assert resp.status_code in (200, 404)
