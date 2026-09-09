"""FastAPI cho production hybrid movie retrieval."""

import logging
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from retrieval.config import settings
from retrieval.service import SearchService
from retrieval.store import check_readiness

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MovieScout Hybrid Retrieval API",
    description=(
        "Tìm kiếm phim bằng BM25 và dense retrieval, RRF fusion, "
        "Cross-Encoder reranking và bộ lọc metadata. V1 hỗ trợ query tiếng Anh."
    ),
    version="2.0.0",
)


class SearchRequest(BaseModel):
    """Request contract của endpoint search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Mô tả nội dung phim bằng tiếng Anh.",
        json_schema_extra={"example": "A father communicates with his daughter through a black hole"},
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=settings.rerank_k,
        description=f"Số kết quả trả về (1-{settings.rerank_k}).",
    )
    genre: str = Field(default="", max_length=80, description="Genre exact-match, bỏ trống để tìm tất cả.")
    year: str = Field(default="", max_length=20, description="Năm hoặc khoảng năm, ví dụ 2010-2020.")
    debug: bool = Field(default=False, description="Expose retrieval evidence nội bộ để debug.")


class MovieItemResponse(BaseModel):
    """Movie metadata và điểm rerank của một kết quả."""

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
    rerank_score: float = Field(description="Điểm Cross-Encoder, không phải xác suất.")
    display_score: float = Field(
        description="Điểm hiển thị tương đối trong chính result set, không dùng để so sánh query."
    )
    evidence: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    """Response contract ổn định cho UI và client."""

    query: str
    results: list[MovieItemResponse]
    index_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Liveness response, không phụ thuộc Qdrant."""

    status: str = "ok"
    service: str = "MovieScout Hybrid Retrieval"


class ReadyResponse(BaseModel):
    """Readiness response sau khi kiểm tra model/index contract."""

    status: str
    index_version: str
    collection: str
    point_count: int


@lru_cache(maxsize=1)
def get_service() -> SearchService:
    """Tạo SearchService một lần cho process API."""

    return SearchService()


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness probe của process."""

    return HealthResponse()


@app.get("/ready", response_model=ReadyResponse, tags=["System"])
def ready() -> ReadyResponse:
    """Readiness probe: Qdrant, alias, manifest, count và dimension phải hợp lệ."""

    try:
        return ReadyResponse(**check_readiness())
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Readiness check gặp lỗi không xác định")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dịch vụ tìm kiếm gặp lỗi nội bộ.",
        ) from exc


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    responses={
        422: {"description": "Tham số truy vấn không hợp lệ"},
        503: {"description": "Qdrant hoặc index contract chưa sẵn sàng"},
        500: {"description": "Lỗi nội bộ không tiết lộ stack trace"},
    },
)
def search(request: SearchRequest) -> SearchResponse:
    """Chạy Dense/BM25 song song, RRF, Cross-Encoder rồi trả top-N."""

    try:
        # Lấy service sau khi Pydantic đã validate request để lỗi 422 không tải model nặng.
        service = get_service()
        return SearchResponse(
            **service.search(
                query=request.query,
                top_n=request.top_n,
                genre=request.genre,
                year=request.year,
                debug=request.debug,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search request gặp lỗi không xác định")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dịch vụ tìm kiếm gặp lỗi nội bộ.",
        ) from exc
