"""Tầng truy cập Qdrant cho alias index hiện tại."""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from .config import settings
from .manifest import load_manifest, validate_manifest

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


def _collection_dense_size(collection_info: Any) -> int | None:
    """Đọc số chiều dense từ các dạng response Qdrant khác nhau."""

    try:
        vectors = collection_info.config.params.vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else vectors
        return int(dense.size)
    except (AttributeError, TypeError, ValueError):
        return None


def validate_index_contract(client: Any | None = None) -> dict[str, Any]:
    """Kiểm tra manifest, alias, point count và dense dimension trước khi query."""

    manifest = load_manifest()
    validate_manifest(manifest)
    qdrant = client or get_client()
    try:
        aliases = qdrant.get_aliases().aliases
        active_collection = next(
            (item.collection_name for item in aliases if item.alias_name == settings.index_alias),
            None,
        )
        if active_collection != manifest["collection"]:
            raise RuntimeError(
                f"Alias {settings.index_alias!r} đang trỏ tới {active_collection!r}, "
                f"manifest yêu cầu {manifest['collection']!r}."
            )
        collection_info = qdrant.get_collection(settings.index_alias)
        count = int(qdrant.count(collection_name=settings.index_alias, exact=True).count)
    except Exception as exc:
        raise RuntimeError("Alias/index Qdrant chưa sẵn sàng.") from exc

    if count != int(manifest["point_count"]):
        raise RuntimeError(
            "Point count của index không khớp manifest: "
            f"expected={manifest['point_count']}, actual={count}."
        )
    dense_size = _collection_dense_size(collection_info)
    if dense_size is not None and dense_size != settings.dense_dimension:
        raise RuntimeError(
            "Dense dimension của index không khớp model: "
            f"expected={settings.dense_dimension}, actual={dense_size}."
        )
    if manifest["collection"] == settings.index_alias:
        raise RuntimeError("Manifest phải trỏ tới collection versioned, không phải alias phục vụ traffic.")
    return manifest


def check_readiness() -> dict[str, Any]:
    """Trả thông tin readiness khi toàn bộ model/index contract hợp lệ."""

    manifest = validate_index_contract()
    return {
        "status": "ready",
        "index_version": manifest["index_version"],
        "collection": manifest["collection"],
        "point_count": manifest["point_count"],
    }


def _query(
    vector: Any,
    vector_name: str,
    limit: int,
    query_filter: Any | None = None,
    *,
    check_contract: bool = True,
) -> list[dict[str, Any]]:
    """Query một vector space trên alias đã được kiểm tra contract."""

    if limit <= 0:
        return []
    if check_contract:
        validate_index_contract()
    response = get_client().query_points(
        collection_name=settings.index_alias,
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
    *,
    check_contract: bool = True,
) -> list[dict[str, Any]]:
    """Tìm ứng viên theo dense vector."""

    return _query(
        vector,
        "dense",
        settings.retrieval_k if limit is None else limit,
        query_filter,
        check_contract=check_contract,
    )


def sparse_search(
    vector: tuple[list[int], list[float]],
    query_filter: Any | None = None,
    limit: int | None = None,
    *,
    check_contract: bool = True,
) -> list[dict[str, Any]]:
    """Tìm ứng viên theo sparse BM25 vector."""

    from qdrant_client import models

    sparse = models.SparseVector(indices=vector[0], values=vector[1])
    return _query(
        sparse,
        "sparse",
        settings.retrieval_k if limit is None else limit,
        query_filter,
        check_contract=check_contract,
    )


def hybrid_search(
    dense_vector: list[float],
    sparse_vector: tuple[list[int], list[float]],
    query_filter: Any | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chạy song song hai nhánh, cho phép graceful degradation một nhánh."""

    search_limit = settings.retrieval_k if limit is None else limit
    if search_limit <= 0:
        return [], []

    # Verify một lần trước khi fan-out để không nhân đôi request readiness.
    validate_index_contract()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            "dense": pool.submit(
                dense_search,
                dense_vector,
                query_filter,
                search_limit,
                check_contract=False,
            ),
            "sparse": pool.submit(
                sparse_search,
                sparse_vector,
                query_filter,
                search_limit,
                check_contract=False,
            ),
        }
        results: dict[str, list[dict[str, Any]]] = {"dense": [], "sparse": []}
        errors: list[Exception] = []
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                logger.exception("Nhánh retrieval %s gặp lỗi", name)
                errors.append(exc)

    if len(errors) == 2:
        raise RuntimeError("Không thể truy vấn Qdrant ở cả hai nhánh dense và sparse.") from errors[0]
    return results["dense"], results["sparse"]
