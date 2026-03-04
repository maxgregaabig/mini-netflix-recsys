"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.common.config import get_serving_config
from src.common.logging import setup_logging
from src.serving.dependencies import get_batcher, init_dependencies
from src.serving.middleware import BackpressureMiddleware, ObservabilityMiddleware
from src.serving.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle."""
    setup_logging()
    config = get_serving_config()
    init_dependencies(config)
    batcher = get_batcher()
    await batcher.start()
    yield
    await batcher.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mini-Netflix Recommender API",
        version="0.1.0",
        lifespan=lifespan,
    )
    config = get_serving_config()
    bp_cfg = config.get("backpressure", {})
    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(BackpressureMiddleware, max_concurrent=bp_cfg.get("max_concurrent_requests", 50))
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = get_serving_config()
    uvicorn.run(
        "src.serving.app:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        workers=config["server"]["workers"],
        reload=False,
    )
