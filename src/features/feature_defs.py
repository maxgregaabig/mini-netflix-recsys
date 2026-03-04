"""Feature definitions and schema constants."""
from __future__ import annotations

# All genres in MovieLens
ALL_GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "(no genres listed)",
]

GENRE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(ALL_GENRES)}

USER_FEATURE_COLS: list[str] = [
    "user_id",
    "user_watch_count_30d",
    "user_genre_affinity",
    "user_mean_rating",
    "user_recency_mean_days",
]

ITEM_FEATURE_COLS: list[str] = [
    "item_id",
    "item_watch_count",
    "genres",
    "genres_multihot",
    "item_mean_rating",
    "title",
]

RANKING_FEATURE_COLS: list[str] = [
    "retrieval_score",
    "item_watch_count",
    "user_mean_rating",
    "item_mean_rating",
    "genre_overlap_score",
    "user_recency_mean_days",
]
