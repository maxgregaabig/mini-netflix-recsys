"""FastAPI middleware for logging, tracing, timeouts, and backpressure."""
from __future__ import annotations

import asyncio
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.common.logging import generate_request_id, get_logger
from src.common.metrics import REQUEST_COUNT, REQUEST_LATENCY

logger = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Adds request_id, structured logging, and latency metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = generate_request_id()
        request.state.request_id = request_id
        request.state.start_time = time.monotonic()

        try:
            response = await call_next(request)
            elapsed = time.monotonic() - request.state.start_time

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
            ).inc()
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(elapsed)

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.1f}"

            logger.info(
                "request_complete",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=round(elapsed * 1000, 1),
            )
            return response

        except Exception as e:
            elapsed = time.monotonic() - request.state.start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500,
            ).inc()
            logger.error(
                "request_error",
                request_id=request_id,
                error=str(e),
                latency_ms=round(elapsed * 1000, 1),
            )
            raise


class BackpressureMiddleware(BaseHTTPMiddleware):
    """Reject requests when the server is overloaded."""

    def __init__(self, app, max_concurrent: int = 50) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip backpressure for health/metrics
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        if self._semaphore._value <= 0:  # noqa: SLF001
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=429,
            ).inc()
            return Response(
                content='{"error":"too_many_requests","detail":"Server is overloaded, please retry later"}',
                status_code=429,
                media_type="application/json",
            )

        async with self._semaphore:
            return await call_next(request)
