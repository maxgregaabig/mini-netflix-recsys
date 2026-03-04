"""Standalone async load test script (no Locust dependency)."""
from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

from src.common.config import REPORTS_DIR, ensure_dirs

API_URL = "http://localhost:8000"
NUM_REQUESTS = 200
CONCURRENCY = 20


async def send_request(client: httpx.AsyncClient, results: list[dict]) -> None:
    user_id = random.randint(1, 610)
    url = f"{API_URL}/recs/home?user_id={user_id}"

    start = time.monotonic()
    try:
        resp = await client.get(url, timeout=10)
        elapsed = (time.monotonic() - start) * 1000
        results.append({
            "status": resp.status_code,
            "latency_ms": round(elapsed, 2),
            "success": resp.status_code == 200,
        })
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        results.append({
            "status": 0,
            "latency_ms": round(elapsed, 2),
            "success": False,
            "error": str(e),
        })


async def run_load_test() -> dict:
    ensure_dirs()
    results: list[dict] = []

    print(f"Running load test: {NUM_REQUESTS} requests, {CONCURRENCY} concurrent")
    start = time.monotonic()

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded_request() -> None:
            async with sem:
                await send_request(client, results)

        tasks = [asyncio.create_task(bounded_request()) for _ in range(NUM_REQUESTS)]
        await asyncio.gather(*tasks)

    total_time = time.monotonic() - start

    # Compute stats
    latencies = [r["latency_ms"] for r in results if r["success"]]
    successes = sum(1 for r in results if r["success"])
    errors = sum(1 for r in results if not r["success"])

    report = {
        "total_requests": NUM_REQUESTS,
        "concurrency": CONCURRENCY,
        "total_time_s": round(total_time, 2),
        "throughput_rps": round(NUM_REQUESTS / total_time, 1),
        "successes": successes,
        "errors": errors,
        "error_rate": round(errors / NUM_REQUESTS * 100, 1),
    }

    if latencies:
        latencies.sort()
        report["latency"] = {
            "min_ms": round(min(latencies), 1),
            "p50_ms": round(latencies[len(latencies) // 2], 1),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
            "max_ms": round(max(latencies), 1),
            "mean_ms": round(statistics.mean(latencies), 1),
        }

    report_path = REPORTS_DIR / "loadtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Load Test Results ---")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_path}")
    return report


if __name__ == "__main__":
    asyncio.run(run_load_test())
