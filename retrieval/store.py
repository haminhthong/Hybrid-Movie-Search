"""Tầng truy cập Qdrant cho Dense và Sparse retrieval."""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> Any:
    """Tạo một Qdrant client dùng chung trong process."""
    from qdrant_client import QdrantClient

    settings.require_qdrant()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30.0,
    )


def check_collection_exists(client: Any | None = None) -> None:
    """Kiểm tra collection đã được khởi tạo hay chưa."""
    qdrant = client or get_client()
    if not qdrant.collection_exists(settings.collection_name):
        raise RuntimeError(
            f"Collection Qdrant '{settings.collection_name}' không tồn tại. "
            "Vui lòng chạy 'python -m scripts.build_index' trước khi tìm kiếm."
        )


def _query(
    vector: Any,
    vector_name: str,
    limit: int,
    query_filter: Any | None = None,
) -> list[dict[str, Any]]:
    """Query một vector space trên Qdrant collection."""
    if limit <= 0:
        return []

    response = get_client().query_points(
        collection_name=settings.collection_name,
        using=vector_name,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {
            "id": str(point.id),
            "score": float(point.score),
            "payload": dict(point.payload or {}),
        }
        for point in response.points
    ]


def dense_search(
    vector: list[float],
    query_filter: Any | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Tìm ứng viên theo dense vector."""
    return _query(
        vector,
        "dense",
        settings.retrieval_k if limit is None else limit,
        query_filter,
    )


def sparse_search(
    vector: tuple[list[int], list[float]],
    query_filter: Any | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Tìm ứng viên theo sparse BM25 vector."""
    from qdrant_client import models

    sparse = models.SparseVector(indices=vector[0], values=vector[1])
    return _query(
        sparse,
        "sparse",
        settings.retrieval_k if limit is None else limit,
        query_filter,
    )


def hybrid_search(
    dense_vector: list[float],
    sparse_vector: tuple[list[int], list[float]],
    query_filter: Any | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chạy song song hai nhánh Dense và Sparse trên Qdrant."""
    search_limit = settings.retrieval_k if limit is None else limit
    if search_limit <= 0:
        return [], []

    check_collection_exists()

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_dense = pool.submit(dense_search, dense_vector, query_filter, search_limit)
        future_sparse = pool.submit(sparse_search, sparse_vector, query_filter, search_limit)

        dense_results = future_dense.result()
        sparse_results = future_sparse.result()

    return dense_results, sparse_results
