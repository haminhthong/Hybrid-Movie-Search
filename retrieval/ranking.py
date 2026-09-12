"""Rank fusion và chuyển payload Qdrant thành movie document có cấu trúc."""

import json
from collections import defaultdict
from typing import Any

from .config import settings


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    limit: int | None = None,
    rrf_k: int | None = None,
) -> list[dict[str, Any]]:
    """Gộp hai danh sách bằng Reciprocal Rank Fusion (RRF) với thứ hạng 1-based.

    Score RRF = sum(1 / (k + rank)) cho từng nhánh.
    """
    max_candidates = settings.rerank_k if limit is None else limit
    if max_candidates <= 0:
        return []
    smooth = settings.rrf_k if rrf_k is None else rrf_k
    if smooth < 0:
        raise ValueError("rrf_k không được âm.")

    scores: defaultdict[str, float] = defaultdict(float)
    documents: dict[str, dict[str, Any]] = {}
    branch_ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for branch_name, results in (("dense", dense), ("sparse", sparse)):
        for rank, item in enumerate(results, start=1):
            document_id = str(item["id"])
            scores[document_id] += 1.0 / (smooth + rank)
            branch_ranks[document_id][branch_name] = rank
            documents.setdefault(document_id, item)

    ranked_ids = sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            branch_ranks[document_id].get("dense", 10**9),
            branch_ranks[document_id].get("sparse", 10**9),
            document_id,
        ),
    )
    fused = []
    for rrf_rank, document_id in enumerate(ranked_ids[:max_candidates], start=1):
        fused.append(
            {
                **documents[document_id],
                "score": scores[document_id],
                "rrf_score": scores[document_id],
                "rrf_rank": rrf_rank,
                "dense_rank": branch_ranks[document_id].get("dense"),
                "sparse_rank": branch_ranks[document_id].get("sparse"),
            }
        )
    return fused


def _as_list(value: Any) -> list[str]:
    """Đọc mảng metadata từ JSON hoặc list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _as_int(value: Any, default: int = 0) -> int:
    """Chuyển số nguyên từ payload một cách an toàn."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    """Chuyển số thực từ payload một cách an toàn."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_movies(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chuyển payload Qdrant thành movie document có metadata có cấu trúc."""
    movies: list[dict[str, Any]] = []
    seen: set[Any] = set()

    for document in documents:
        payload = document.get("payload") or {}
        movie_id = payload.get("movie_id")
        if movie_id is None:
            continue
        normalized_movie_id = _as_int(movie_id)
        if normalized_movie_id <= 0 or normalized_movie_id in seen:
            continue
        seen.add(normalized_movie_id)

        movies.append(
            {
                "movie_id": normalized_movie_id,
                "title": str(payload.get("title") or "Không rõ tên"),
                "director": str(payload.get("director") or ""),
                "cast": _as_list(payload.get("cast")),
                "genres": _as_list(payload.get("genres")),
                "keywords": _as_list(payload.get("keywords")),
                "overview": str(payload.get("overview") or ""),
                "release_date": str(payload.get("release_date") or ""),
                "release_year": _as_int(payload.get("release_year")),
                "vote_average": _as_float(payload.get("vote_average")),
                "popularity": _as_float(payload.get("popularity")),
                "poster_path": str(payload.get("poster_path") or ""),
                "document": {
                    "id": str(document["id"]),
                    "text": str(payload.get("document_text") or ""),
                },
                "rrf_score": float(document.get("rrf_score", document.get("score", 0.0))),
                "dense_rank": document.get("dense_rank"),
                "sparse_rank": document.get("sparse_rank"),
                "rrf_rank": document.get("rrf_rank"),
            }
        )
    return movies


def rank_movies(movies: list[dict[str, Any]], top_n: int = 10) -> list[dict[str, Any]]:
    """Sắp xếp danh sách phim theo điểm rerank hoặc RRF và gán rank 1-based."""
    if not movies or top_n <= 0:
        return []

    ranked = sorted(
        movies,
        key=lambda movie: (
            -float(movie.get("rerank_score", movie.get("rrf_score", 0.0))),
            int(movie.get("rrf_rank") or 10**9),
            int(movie.get("movie_id") or 0),
        ),
    )
    for rank, movie in enumerate(ranked[:top_n], start=1):
        movie["rank"] = rank
    return ranked[:top_n]
