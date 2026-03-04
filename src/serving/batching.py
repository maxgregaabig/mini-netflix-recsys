"""Dynamic batching for LightGBM ranking inference.

Collects scoring requests over a configurable time window, merges them into a
single batch call, then dispatches per-request results.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.logging import get_logger
from src.common.metrics import BATCH_SIZE, QUEUE_DEPTH

logger = get_logger(__name__)


@dataclass
class ScoringRequest:
    """A single ranking scoring request."""
    request_id: str
    features: np.ndarray         # shape (N_candidates, n_features)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    submitted_at: float = field(default_factory=time.monotonic)


class DynamicBatcher:
    """Collects ranking requests and batches them for efficient scoring.

    Non-blocking and concurrency-safe via asyncio.
    """

    def __init__(
        self,
        score_fn: Any,           # callable: np.ndarray -> np.ndarray
        max_wait_ms: float = 10,
        max_batch_size: int = 64,
        max_queue_depth: int = 100,
    ):
        self._score_fn = score_fn
        self._max_wait = max_wait_ms / 1000.0
        self._max_batch_size = max_batch_size
        self._max_queue_depth = max_queue_depth
        self._queue: asyncio.Queue[ScoringRequest] = asyncio.Queue(maxsize=max_queue_depth)
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background batch processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("DynamicBatcher started")

    async def stop(self) -> None:
        """Stop the batcher."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def submit(self, request_id: str, features: np.ndarray, timeout: float = 2.0) -> np.ndarray:
        """Submit a scoring request and wait for the result.

        Args:
            request_id: Unique request identifier
            features: Feature matrix for candidates (N, F)
            timeout: Max seconds to wait for result

        Returns:
            Scores array (N,)

        Raises:
            asyncio.QueueFull: If queue is at max depth (triggers 429)
            asyncio.TimeoutError: If scoring takes too long
        """
        req = ScoringRequest(request_id=request_id, features=features)

        try:
            self._queue.put_nowait(req)
        except asyncio.QueueFull:
            raise

        QUEUE_DEPTH.set(self._queue.qsize())
        return await asyncio.wait_for(req.future, timeout=timeout)

    async def _process_loop(self) -> None:
        """Main loop: collect requests, batch, score, dispatch."""
        while self._running:
            batch: list[ScoringRequest] = []

            # Wait for first request
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                batch.append(first)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                continue

            # Collect more within the time window
            deadline = time.monotonic() + self._max_wait
            while len(batch) < self._max_batch_size and time.monotonic() < deadline:
                try:
                    remaining = max(deadline - time.monotonic(), 0.001)
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            QUEUE_DEPTH.set(self._queue.qsize())
            BATCH_SIZE.observe(len(batch))

            # Merge feature matrices and score
            try:
                offsets = []
                all_features = []
                for req in batch:
                    offsets.append(len(req.features))
                    all_features.append(req.features)

                merged = np.concatenate(all_features, axis=0)
                all_scores = self._score_fn(merged)

                # Dispatch results
                idx = 0
                for req, n in zip(batch, offsets):
                    scores = all_scores[idx:idx + n]
                    idx += n
                    if not req.future.done():
                        req.future.set_result(scores)

            except Exception as e:
                logger.error("Batch scoring error", error=str(e))
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)
