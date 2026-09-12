"""Two-stage hybrid retrieval orchestrator: Dense + BM25 → RRF → Cross-Encoder."""

import logging
import re
import time
from typing import Any

from .config import settings
from .query import QueryEncoder
from .ranking import rank_movies, reciprocal_rank_fusion, to_movies
from .rerank import CrossEncoderReranker
from .store import hybrid_search

logger = logging.getLogger(__name__)


def parse_year(value: str) -> tuple[int, int]:
    """Phân tích năm đơn hoặc khoảng năm (ví dụ: 2014 hoặc 2010-2020)."""
    if not isinstance(value, str):
        raise TypeError("Định dạng năm phải là chuỗi ký tự.")
    match = re.fullmatch(r"\s*(\d{4})(?:\s*(?:-|to)\s*(\d{4}))?\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError("Định dạng năm không hợp lệ. Dùng YYYY hoặc YYYY-YYYY, ví dụ 2010-2020.")

    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    if start > end:
        raise ValueError("Năm bắt đầu không được lớn hơn năm kết thúc.")
    if not 1888 <= start <= end <= 2100:
        raise ValueError("Khoảng năm phải nằm trong từ năm 1888 đến 2100.")
    return start, end


def build_filter(genre: str = "", year: str = "") -> Any | None:
    """Tạo filter genre exact-match không phân biệt hoa thường và year range."""
    normalized_genre = genre.strip()
    normalized_year = year.strip()
    filter_values: list[tuple[str, Any]] = []
    if normalized_genre and normalized_genre.casefold() != "all":
        filter_values.append(("genre", normalized_genre.casefold()))

    if normalized_year:
        filter_values.append(("year", parse_year(normalized_year)))
    if not filter_values:
        return None

    from qdrant_client import models

    conditions: list[Any] = []
    for field, value in filter_values:
        if field == "genre":
            conditions.append(
                models.FieldCondition(
                    key="genre_keys",
                    match=models.MatchValue(value=value),
                )
            )
        else:
            start, end = value
            conditions.append(
                models.FieldCondition(
                    key="release_year",
                    range=models.Range(gte=start, lte=end),
                )
            )

    return models.Filter(must=conditions) if conditions else None


class MovieSearch:
    """Pipeline Hybrid Movie Retrieval: Dense + BM25 -> RRF -> Cross-Encoder."""

    def __init__(
        self,
        encoder: QueryEncoder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.encoder = encoder or QueryEncoder()
        self.reranker = reranker

    def _candidates(
        self,
        dense_vector: list[float],
        sparse_vector: tuple[list[int], list[float]],
        query_filter: Any | None,
    ) -> list[dict[str, Any]]:
        """Thu thập candidate từ hai nhánh Dense và Sparse, sau đó fusion bằng RRF."""
        dense, sparse = hybrid_search(
            dense_vector,
            sparse_vector,
            query_filter,
            limit=settings.retrieval_k,
        )
        fused = reciprocal_rank_fusion(
            dense,
            sparse,
            limit=settings.rerank_k,
            rrf_k=settings.rrf_k,
        )
        return to_movies(fused)

    def _get_reranker(self) -> CrossEncoderReranker:
        """Nạp lười Cross-Encoder reranker."""
        if self.reranker is None:
            self.reranker = CrossEncoderReranker()
        return self.reranker

    @staticmethod
    def _public_movie(movie: dict[str, Any], debug: bool = False) -> dict[str, Any]:
        """Tạo đối tượng movie gọn gàng cho public response."""
        public_fields = (
            "movie_id",
            "title",
            "director",
            "cast",
            "genres",
            "keywords",
            "overview",
            "release_date",
            "release_year",
            "vote_average",
            "popularity",
            "poster_path",
            "rank",
        )
        result = {field: movie[field] for field in public_fields if field in movie}
        if "rerank_score" in movie:
            result["rerank_score"] = movie["rerank_score"]
        if debug:
            result["evidence"] = {
                "dense_rank": movie.get("dense_rank"),
                "sparse_rank": movie.get("sparse_rank"),
                "rrf_rank": movie.get("rrf_rank"),
                "rrf_score": movie.get("rrf_score"),
                "rerank_score": movie.get("rerank_score"),
            }
        return result

    def search(
        self,
        query: str,
        top_n: int = 10,
        genre: str = "",
        year: str = "",
        debug: bool = False,
    ) -> dict[str, Any]:
        """Tìm kiếm phim qua pipeline 2-stage hybrid retrieval."""
        started = time.perf_counter()

        clean_query = QueryEncoder.clean_query(query)
        if not clean_query:
            raise ValueError("Truy vấn tìm kiếm không được để trống hoặc chỉ chứa ký tự đặc biệt.")
        if not 0 < top_n <= settings.rerank_k:
            raise ValueError(f"top_n phải nằm trong khoảng 1-{settings.rerank_k}.")

        query_filter = build_filter(genre, year)
        clean_query, dense_vector, sparse_vector = self.encoder.encode(query)
        candidates = self._candidates(dense_vector, sparse_vector, query_filter)

        if not candidates:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return {"query": clean_query, "results": [], "latency_ms": latency_ms}

        # Stage 2: Cross-Encoder reranking trên candidate set nhỏ
        reranked = self._get_reranker().rerank(clean_query, candidates[: settings.rerank_k])
        results = rank_movies(reranked, top_n=top_n)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Search thành công: query='%s', results=%d, latency=%.2f ms",
            clean_query,
            len(results),
            latency_ms,
        )
        return {
            "query": clean_query,
            "results": [self._public_movie(movie, debug) for movie in results],
            "latency_ms": latency_ms,
        }
