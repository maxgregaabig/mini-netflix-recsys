"""Pydantic schemas for API request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class MovieRec(BaseModel):
    item_id: int
    title: str = ""
    score: float
    genres: list[str] = Field(default_factory=list)


class RecRow(BaseModel):
    row_name: str
    items: list[MovieRec]


class HomeRecsResponse(BaseModel):
    user_id: int
    rows: list[RecRow]
    latency_ms: float


class EventPayload(BaseModel):
    user_id: int
    item_id: int
    event_type: str = "watch"
    watch_time: float = 0.0
    rating: float | None = None


class EventResponse(BaseModel):
    status: str = "logged"


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
