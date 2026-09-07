"""Test metric hierarchy cho graded relevance."""

from evaluation.metrics import evaluate_ranking, ndcg_at_k
from evaluation.errors import rerank_gain_harm


def test_ndcg_rewards_graded_relevance():
    judgments = {"a": 3, "b": 2, "c": 1}
    assert ndcg_at_k(["a", "b", "c"], judgments, 3) == 1.0
    assert ndcg_at_k(["c", "b", "a"], judgments, 3) < 1.0


def test_evaluation_returns_primary_and_candidate_metrics():
    metrics = evaluate_ranking(["x", "b", "a"], {"a": 3, "b": 2})
    assert set(("ndcg@10", "mrr@10", "recall@10", "recall@50")).issubset(metrics)
    assert metrics["mrr@10"] == 0.5


def test_rerank_gain_harm_is_reported():
    before = [{"movie_id": 1}, {"movie_id": 2}, {"movie_id": 3}]
    after = [{"movie_id": 2}, {"movie_id": 1}, {"movie_id": 3}]
    stats = rerank_gain_harm(before, after, {"1", "3"})
    assert stats["queries_improved"] == 0.0
    assert stats["queries_harmed"] == 1.0
