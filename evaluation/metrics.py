"""Metric IR cho known-item và thematic multi-relevance queries."""

import math
from typing import Iterable


def dcg_at_k(relevances: Iterable[int | float], k: int = 10) -> float:
    """Tính DCG dùng gain graded ``2^rel - 1``."""

    if k <= 0:
        return 0.0
    return sum(
        (2**float(relevance) - 1) / math.log2(position + 2)
        for position, relevance in enumerate(list(relevances)[:k])
    )


def ndcg_at_k(retrieved_ids: list[str], judgments: dict[str, int], k: int = 10) -> float:
    """Tính nDCG@k cho relevance judgments graded."""

    actual = [judgments.get(str(movie_id), 0) for movie_id in retrieved_ids[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal, k)
    return 0.0 if ideal_dcg == 0 else dcg_at_k(actual, k) / ideal_dcg


def reciprocal_rank_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Tính MRR@k cho query có một hoặc nhiều positive."""

    for position, movie_id in enumerate(retrieved_ids[:k], start=1):
        if str(movie_id) in relevant_ids:
            return 1.0 / position
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Tính Recall@k trên toàn bộ relevant set."""

    if not relevant_ids:
        return 0.0
    found = relevant_ids.intersection({str(movie_id) for movie_id in retrieved_ids[:k]})
    return len(found) / len(relevant_ids)


def evaluate_ranking(
    retrieved_ids: list[str],
    judgments: dict[str, int],
    candidate_ids: list[str] | None = None,
) -> dict[str, float]:
    """Tính metric hierarchy; Recall@50 có thể dùng pool trước rerank."""

    relevant_ids = {movie_id for movie_id, grade in judgments.items() if grade > 0}
    candidate_ranking = retrieved_ids if candidate_ids is None else candidate_ids
    return {
        "ndcg@10": ndcg_at_k(retrieved_ids, judgments, 10),
        "mrr@10": reciprocal_rank_at_k(retrieved_ids, relevant_ids, 10),
        "recall@10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "recall@50": recall_at_k(candidate_ranking, relevant_ids, 50),
        "hit@1": float(bool(retrieved_ids and str(retrieved_ids[0]) in relevant_ids)),
        "hit@5": float(bool(set(map(str, retrieved_ids[:5])) & relevant_ids)),
    }
