"""Regression benchmark legacy cho BM25 và dense trên synthetic query set.

Tập này dùng để bắt regression/sanity check; metric headline của project nằm ở
``evaluation/benchmark.py`` với graded judgments trên Dev và Locked Test.
"""

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from retrieval.query import QueryEncoder
from retrieval.ranking import to_movies
from retrieval.store import dense_search, sparse_search

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_FILE = PROJECT_ROOT / "evaluation" / "eval_queries_200.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "reports" / "synthetic"
METRIC_NAMES = ("hit@1", "hit@3", "hit@5", "hit@10", "mrr@10")
REQUIRED_COLUMNS = {"query_id", "query", "query_type", "expected_movie_id", "expected_title"}


class SimpleRetriever:
    """Mã hóa truy vấn một lần rồi chạy hai baseline độc lập."""

    def __init__(self, top_n: int = 10):
        self.query_encoder = QueryEncoder()
        self.top_n = top_n

    def retrieve_both(self, query: str) -> dict[str, list[dict]]:
        """Trả kết quả BM25 và dense cho cùng một truy vấn."""
        _, dense_vector, sparse_vector = self.query_encoder.encode(query)
        return {
            "bm25": to_movies(sparse_search(sparse_vector))[: self.top_n],
            "dense": to_movies(dense_search(dense_vector))[: self.top_n],
        }


def calculate_metrics(results: list[dict], expected_movie_id: int) -> dict[str, float]:
    """Tính Hit@K và reciprocal rank trong top 10."""
    retrieved_ids = [int(item["movie_id"]) for item in results[:10]]
    reciprocal_rank = 0.0
    if expected_movie_id in retrieved_ids:
        reciprocal_rank = 1.0 / (retrieved_ids.index(expected_movie_id) + 1)
    return {
        "hit@1": float(expected_movie_id in retrieved_ids[:1]),
        "hit@3": float(expected_movie_id in retrieved_ids[:3]),
        "hit@5": float(expected_movie_id in retrieved_ids[:5]),
        "hit@10": float(expected_movie_id in retrieved_ids[:10]),
        "mrr@10": reciprocal_rank,
    }


def _validate_dataset(dataframe: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Tập đánh giá thiếu các cột: {', '.join(sorted(missing))}")
    if dataframe.empty:
        raise ValueError("Tập đánh giá không có truy vấn.")


def evaluate_batch(
    eval_file: Path = DEFAULT_EVAL_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Chạy benchmark, ghi kết quả chi tiết và tóm tắt dạng văn bản."""
    dataframe = pd.read_csv(eval_file)
    _validate_dataset(dataframe)
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size phải lớn hơn 0.")
        dataframe = dataframe.head(sample_size)

    retriever = SimpleRetriever(top_n=10)
    totals = {method: defaultdict(float) for method in ("bm25", "dense")}
    rows = []
    started = time.perf_counter()
    for position, row in enumerate(dataframe.itertuples(index=False), start=1):
        expected_id = int(row.expected_movie_id)
        method_results = retriever.retrieve_both(str(row.query))
        output = {
            "query_id": row.query_id,
            "query": row.query,
            "query_type": row.query_type,
            "expected_movie_id": expected_id,
            "expected_title": row.expected_title,
        }
        for method, results in method_results.items():
            metrics = calculate_metrics(results, expected_id)
            for name, value in metrics.items():
                totals[method][name] += value
                output[f"{method}_{name}"] = round(value, 4)
            output[f"{method}_top1"] = results[0]["title"] if results else "N/A"
        rows.append(output)
        if position % 20 == 0:
            logger.info("Đã xử lý %d/%d truy vấn", position, len(dataframe))

    elapsed = time.perf_counter() - started
    count = len(dataframe)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame(rows)
    details.to_csv(output_dir / "comparison_results.csv", index=False, encoding="utf-8")

    lines = [
        "SO SÁNH BM25 VÀ DENSE",
        f"Số truy vấn: {count}",
        f"Tổng thời gian: {elapsed:.3f} giây",
        f"Thời gian trung bình: {elapsed / count:.4f} giây/truy vấn",
        "",
        f"{'Chỉ số':<12}{'BM25':>12}{'Dense':>12}",
    ]
    for metric in METRIC_NAMES:
        lines.append(
            f"{metric:<12}{totals['bm25'][metric] / count:>12.4f}{totals['dense'][metric] / count:>12.4f}"
        )
    (output_dir / "comparison_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Đã ghi kết quả benchmark vào %s", output_dir)
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh baseline BM25 và dense")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample", type=int)
    args = parser.parse_args()
    evaluate_batch(args.eval_file, args.output_dir, args.sample)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
