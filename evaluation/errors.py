"""Phân loại retrieval failure theo stage để làm error analysis."""

from typing import Any


FAILURE_STAGES = (
    "NOT_RETRIEVED",
    "FUSION_DROPPED",
    "RERANK_HARM",
    "FILTERED_OUT",
    "CATALOG_MISSING",
)


def classify_failure(
    relevant_ids: set[str],
    *,
    catalog_ids: set[str],
    dense_ids: set[str],
    sparse_ids: set[str],
    fused_ids: set[str],
    reranked_ids: set[str],
    filters_applied: bool = False,
) -> str | None:
    """Xác định stage đầu tiên làm mất relevant item."""

    if not relevant_ids.intersection(catalog_ids):
        return "CATALOG_MISSING"
    if filters_applied and not relevant_ids.intersection(dense_ids | sparse_ids):
        return "FILTERED_OUT"
    if not relevant_ids.intersection(dense_ids | sparse_ids):
        return "NOT_RETRIEVED"
    if not relevant_ids.intersection(fused_ids):
        return "FUSION_DROPPED"
    if not relevant_ids.intersection(reranked_ids):
        return "RERANK_HARM"
    return None


def rank_map(items: list[dict[str, Any]]) -> dict[str, int]:
    """Tạo map movie_id → rank từ output retrieval."""

    return {
        str(item.get("movie_id")): rank
        for rank, item in enumerate(items, start=1)
        if item.get("movie_id") is not None
    }


def rerank_gain_harm(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    relevant_ids: set[str],
) -> dict[str, float]:
    """Đo số query/item được reranker cải thiện hoặc làm xấu đi."""

    before_ranks = rank_map(before)
    after_ranks = rank_map(after)
    deltas = [
        after_ranks[movie_id] - before_ranks[movie_id]
        for movie_id in relevant_ids
        if movie_id in before_ranks and movie_id in after_ranks
    ]
    return {
        "queries_improved": float(sum(delta < 0 for delta in deltas)),
        "queries_harmed": float(sum(delta > 0 for delta in deltas)),
        "mean_rank_delta": sum(deltas) / len(deltas) if deltas else 0.0,
    }
