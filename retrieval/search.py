"""Orchestrator của production retrieval: Dense + BM25 → RRF → Cross-Encoder."""

import re
from typing import Any

from .config import settings
from .query import QueryEncoder
from .ranking import add_display_scores, reciprocal_rank_fusion, to_movies
from .rerank import CrossEncoderReranker
from .store import hybrid_search


def parse_year(value: str) -> tuple[int, int]:
    """Phân tích năm đơn hoặc khoảng năm trong khoảng hợp lệ của dữ liệu phim."""

    if not isinstance(value, str):
        raise ValueError("Định dạng năm không hợp lệ. Dùng YYYY hoặc YYYY-YYYY, ví dụ 2010-2020.")
    match = re.fullmatch(r"\s*(\d{4})(?:\s*(?:-|to)\s*(\d{4}))?\s*", value, re.IGNORECASE)
    if not match:
        raise ValueError(
            "Định dạng năm không hợp lệ. Dùng YYYY hoặc YYYY-YYYY, ví dụ 2010-2020."
        )

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
            # genres trong payload là mảng keyword; MatchValue kiểm tra phần tử chính xác.
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
    """Chạy canonical online retrieval, không route query và không gọi LLM."""

    def __init__(
        self,
        encoder: QueryEncoder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.encoder = encoder or QueryEncoder()
        model_name = getattr(self.encoder, "dense_model_name", None)
        dimension = getattr(self.encoder, "dense_dimension", None)
        if isinstance(model_name, str) and model_name != settings.dense_model:
            raise RuntimeError("QueryEncoder không tương thích với dense model/index contract.")
        if isinstance(dimension, (int, float)) and int(dimension) != settings.dense_dimension:
            raise RuntimeError("QueryEncoder không tương thích với dense model/index contract.")
        self.reranker = reranker

    def _candidates(
        self,
        dense_vector: list[float],
        sparse_vector: tuple[list[int], list[float]],
        query_filter: Any | None,
    ) -> list[dict[str, Any]]:
        """Lấy 50 kết quả mỗi nhánh, fusion rồi giữ 30 candidate đầu."""

        dense, sparse = hybrid_search(
            dense_vector,
            sparse_vector,
            query_filter,
            limit=settings.retrieval_k,
        )
        fused = reciprocal_rank_fusion(
            dense,
            sparse,
            limit=settings.candidate_k,
            rrf_k=settings.rrf_k,
        )
        return to_movies(fused)

    def _get_reranker(self) -> CrossEncoderReranker:
        """Nạp lười Cross-Encoder, nhưng luôn dùng nó trong production path."""

        if self.reranker is None:
            self.reranker = CrossEncoderReranker()
        return self.reranker

    @staticmethod
    def _public_movie(movie: dict[str, Any], debug: bool = False) -> dict[str, Any]:
        """Loại evidence nội bộ khỏi response thông thường."""

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
            "rerank_score",
            "display_score",
        )
        result = {field: movie[field] for field in public_fields if field in movie}
        if debug:
            result["evidence"] = {
                "dense_rank": movie.get("dense_rank"),
                "sparse_rank": movie.get("sparse_rank"),
                "rrf_rank": movie.get("rrf_rank"),
                "rrf_score": movie.get("rrf_score"),
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
        """Tìm phim theo pipeline cố định Dense/BM25 → RRF → Cross-Encoder."""

        if not 0 < top_n <= settings.rerank_k:
            raise ValueError(f"top_n phải nằm trong khoảng 1-{settings.rerank_k}.")

        clean_query, dense_vector, sparse_vector = self.encoder.encode(query)
        candidates = self._candidates(dense_vector, sparse_vector, build_filter(genre, year))
        if not candidates:
            return {"query": clean_query, "results": [], "index_version": settings.index_version}

        # Cross-Encoder chỉ thấy fusion candidates; không rerank toàn bộ catalog.
        reranked = self._get_reranker().rerank(clean_query, candidates[: settings.rerank_k])
        results = add_display_scores(reranked, top_n)
        return {
            "query": clean_query,
            "results": [self._public_movie(movie, debug) for movie in results],
            "index_version": settings.index_version,
        }
