"""Locust load test for the recommendation API."""
from __future__ import annotations

import random

from locust import HttpUser, between, task


class RecsUser(HttpUser):
    """Simulates a user browsing the recommendation API."""

    wait_time = between(0.1, 0.5)
    host = "http://localhost:8000"

    @task(5)
    def get_home_recs(self) -> None:
        user_id = random.randint(1, 610)
        self.client.get(f"/recs/home?user_id={user_id}", name="/recs/home")

    @task(2)
    def get_home_recs_with_seed(self) -> None:
        user_id = random.randint(1, 610)
        seed = random.randint(1, 9742)
        self.client.get(
            f"/recs/home?user_id={user_id}&seed_item_id={seed}",
            name="/recs/home?seed",
        )

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health")

    @task(1)
    def log_event(self) -> None:
        self.client.post("/log_event", json={
            "user_id": random.randint(1, 610),
            "item_id": random.randint(1, 9742),
            "event_type": "watch",
            "watch_time": random.uniform(5, 120),
            "rating": random.choice([3.0, 3.5, 4.0, 4.5, 5.0]),
        })
