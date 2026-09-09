"""Benchmark ablation tối giản B0-B3 trên Dev hoặc Locked Test.

Locked Test chỉ được đọc bởi command final; không có tham số tuning trong module
này để tránh vô tình dùng test set chọn cấu hình.
"""

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from retrieval.config import settings
from retrieval.query import QueryEncoder
from retrieval.ranking import reciprocal_rank_fusion, to_movies
from retrieval.rerank import CrossEncoderReranker
from retrieval.store import hybrid_search

from .errors import rerank_gain_harm
from .judgments import QueryJudgment, load_judgments
from .metrics import evaluate_ranking

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV = PROJECT_ROOT / "evaluation" / "dev_queries.jsonl"
DEFAULT_LOCKED = PROJECT_ROOT / "evaluation" / "locked_test_queries.jsonl"


def _ids(movies: list[dict[str, Any]]) -> list[str]:
    """Lấy movie IDs theo thứ tự ranking."""

    return [str(movie["movie_id"]) for movie in movies]


class AblationRunner:
    """Chạy bốn pipeline trên cùng một lần encode và retrieval."""

    def __init__(self) -> None:
        self.encoder = QueryEncoder()
        self.reranker: CrossEncoderReranker | None = None

    def _get_reranker(self) -> CrossEncoderReranker:
        if self.reranker is None:
            self.reranker = CrossEncoderReranker()
        return self.reranker

    def retrieve(self, judgment: QueryJudgment, pipeline: str) -> list[dict[str, Any]]:
        """Trả ranking của một pipeline B0-B3 (API tương thích cũ)."""

        rankings = self.retrieve_all(judgment)["rankings"]
        if pipeline not in rankings:
            raise ValueError(f"Pipeline không hợp lệ: {pipeline}")
        return rankings[pipeline]

    def retrieve_all(self, judgment: QueryJudgment) -> dict[str, Any]:
        """Tái sử dụng vector và hai branch để benchmark không truy vấn lặp."""

        query, dense_vector, sparse_vector = self.encoder.encode(judgment.query)
        dense, sparse = hybrid_search(dense_vector, sparse_vector, limit=settings.retrieval_k)
        bm25 = to_movies(sparse)
        dense_movies = to_movies(dense)
        fused_all = to_movies(
            reciprocal_rank_fusion(
                dense,
                sparse,
                limit=settings.retrieval_k,
                rrf_k=settings.rrf_k,
            )
        )
        fused_candidates = fused_all[: settings.candidate_k]
        reranked = self._get_reranker().rerank(query, fused_candidates[: settings.rerank_k])
        return {
            "rankings": {
                "B0_BM25": bm25,
                "B1_DENSE": dense_movies,
                "B2_HYBRID_RRF": fused_candidates,
                "B3_HYBRID_CE": reranked,
            },
            "candidate_ids": _ids(fused_all),
        }


def evaluate_split(
    judgments: list[QueryJudgment],
    *,
    output_dir: Path,
) -> pd.DataFrame:
    """Chạy ablation và ghi result theo từng query/pipeline."""

    if not judgments:
        raise ValueError("Không thể đánh giá split rỗng.")
    runner = AblationRunner()
    pipelines = ("B0_BM25", "B1_DENSE", "B2_HYBRID_RRF", "B3_HYBRID_CE")
    rows: list[dict[str, Any]] = []
    totals: dict[str, defaultdict[str, float]] = {
        pipeline: defaultdict(float) for pipeline in pipelines
    }
    rerank_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for judgment in judgments:
        query_started = time.perf_counter()
        retrieval = runner.retrieve_all(judgment)
        rankings: dict[str, list[dict[str, Any]]] = retrieval["rankings"]
        candidate_ids = retrieval["candidate_ids"]
        query_latency_ms = round((time.perf_counter() - query_started) * 1000, 2)
        for pipeline in pipelines:
            movies = rankings[pipeline]
            pool_ids = candidate_ids if pipeline in {"B2_HYBRID_RRF", "B3_HYBRID_CE"} else _ids(movies)
            metrics = evaluate_ranking(_ids(movies), judgment.judgments, candidate_ids=pool_ids)
            row = {
                "query_id": judgment.query_id,
                "query_type": judgment.query_type,
                "pipeline": pipeline,
                "latency_ms": query_latency_ms,
                **metrics,
            }
            rows.append(row)
            for name, value in metrics.items():
                totals[pipeline][name] += value
        rerank_rows.append(
            {
                "query_id": judgment.query_id,
                **rerank_gain_harm(
                    rankings["B2_HYBRID_RRF"],
                    rankings["B3_HYBRID_CE"],
                    judgment.relevant_ids,
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame(rows)
    details.to_csv(output_dir / "benchmark_results.csv", index=False, encoding="utf-8")
    count = len(judgments)
    summary_rows = []
    for pipeline in pipelines:
        values = {name: totals[pipeline][name] / count for name in totals[pipeline]}
        pipeline_rows = details[details["pipeline"] == pipeline]
        summary_rows.append(
            {
                "pipeline": pipeline,
                **{name: round(value, 4) for name, value in values.items()},
                "p50_latency_ms": round(float(pipeline_rows["latency_ms"].quantile(0.50)), 2),
                "p95_latency_ms": round(float(pipeline_rows["latency_ms"].quantile(0.95)), 2),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "benchmark_summary.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(rerank_rows).to_csv(
        output_dir / "rerank_gain_harm.csv",
        index=False,
        encoding="utf-8",
    )
    logger.info(
        "Đánh giá %d query hoàn tất trong %.2f giây: %s",
        count,
        time.perf_counter() - started,
        output_dir,
    )
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá B0-B3 trên một split đã chọn")
    parser.add_argument("--split", choices=("dev", "locked_test"), default="dev")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation" / "reports")
    args = parser.parse_args()
    path = DEFAULT_DEV if args.split == "dev" else DEFAULT_LOCKED
    evaluate_split(load_judgments(path), output_dir=args.output_dir / args.split)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
