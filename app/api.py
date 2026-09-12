"""FastAPI cho Hybrid Movie Retrieval API."""

import logging
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from retrieval.config import settings
from retrieval.search import MovieSearch

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MovieScout Hybrid Retrieval API",
    description="Tìm kiếm phim bằng BM25 và Dense Retrieval, RRF fusion và Cross-Encoder reranking.",
    version="1.0.0",
)


class SearchRequest(BaseModel):
    """Request schema cho endpoint search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Mô tả nội dung hoặc chủ đề phim bằng tiếng Anh.",
        json_schema_extra={"example": "A father communicates with his daughter through a black hole"},
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=settings.rerank_k,
        description=f"Số kết quả trả về (1-{settings.rerank_k}).",
    )
    genre: str = Field(default="", max_length=80, description="Genre exact-match, bỏ trống để tìm tất cả.")
    year: str = Field(default="", max_length=20, description="Năm hoặc khoảng năm, ví dụ 2010 hoặc 2010-2020.")
    debug: bool = Field(default=False, description="Trả về thông tin ranking nội bộ (dense/sparse/RRF) để debug.")


class MovieItemResponse(BaseModel):
    """Movie metadata của một kết quả tìm kiếm."""

    movie_id: int
    title: str
    director: str = ""
    cast: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    overview: str = ""
    release_date: str = ""
    release_year: int = 0
    vote_average: float = 0.0
    popularity: float = 0.0
    poster_path: str = ""
    rank: int
    rerank_score: float | None = None
    evidence: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    """Response schema cho kết quả tìm kiếm."""

    query: str
    results: list[MovieItemResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


@lru_cache(maxsize=1)
def get_search_engine() -> MovieSearch:
    """Khởi tạo MovieSearch một lần cho process API."""
    return MovieSearch()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness probe của dịch vụ."""
    return HealthResponse()


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    responses={
        422: {"description": "Tham số truy vấn không hợp lệ"},
        503: {"description": "Qdrant collection chưa sẵn sàng"},
        500: {"description": "Lỗi nội bộ hệ thống"},
    },
)
def search(request: SearchRequest) -> SearchResponse:
    """Chạy Dense/BM25 retrieval song song, RRF fusion và Cross-Encoder reranking."""
    try:
        engine = get_search_engine()
        result = engine.search(
            query=request.query,
            top_n=request.top_n,
            genre=request.genre,
            year=request.year,
            debug=request.debug,
        )
        return SearchResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search request gặp sự cố không xác định")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dịch vụ tìm kiếm gặp lỗi nội bộ.",
        ) from exc
