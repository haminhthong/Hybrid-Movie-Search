"""Cross-Encoder reranking cho tập ứng viên đã qua RRF."""

import logging
import time
from typing import Any

from .config import RERANK_MODEL

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Xếp hạng chính xác hơn nhưng chỉ trên candidate set nhỏ."""

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Đang nạp Cross-Encoder %s trên %s", model_name, device.upper())
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chấm điểm trực tiếp ``(query, document)`` và sắp xếp giảm dần.

        Search service đảm bảo danh sách truyền vào không vượt quá ``rerank_k``.
        Khi model lỗi, giữ thứ tự RRF để một lỗi model không làm mất kết quả.
        """

        if not movies:
            return []

        pairs = [[query, movie.get("document", {}).get("text", "")] for movie in movies]
        started = time.perf_counter()
        try:
            scores = self.model.predict(pairs)
        except Exception:
            logger.exception("Cross-Encoder lỗi; giữ nguyên thứ tự RRF")
            for movie in movies:
                movie["rerank_score"] = float(movie.get("rrf_score", 0.0))
            return movies

        for position, movie in enumerate(movies):
            fallback = float(movie.get("rrf_score", 0.0))
            movie["rerank_score"] = float(scores[position]) if position < len(scores) else fallback
        elapsed = time.perf_counter() - started
        logger.info("Rerank %d ứng viên trong %.4f giây", len(movies), elapsed)
        return sorted(
            movies,
            key=lambda movie: (
                -float(movie.get("rerank_score", 0.0)),
                int(movie.get("rrf_rank") or 10**9),
                int(movie.get("movie_id") or 0),
            ),
        )
