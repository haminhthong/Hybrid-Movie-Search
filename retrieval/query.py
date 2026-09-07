"""Chuẩn hóa và mã hóa truy vấn cho hai nhánh dense và sparse."""

import logging
import re
from typing import Any

from .config import DENSE_MODEL, SPARSE_MODEL

logger = logging.getLogger(__name__)


class QueryEncoder:
    """Nạp một lần hai encoder dùng bởi online retrieval."""

    def __init__(
        self,
        dense_model_name: str = DENSE_MODEL,
        sparse_model_name: str = SPARSE_MODEL,
    ) -> None:
        from fastembed import SparseTextEmbedding
        from sentence_transformers import SentenceTransformer

        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        logger.info("Đang nạp dense model: %s", dense_model_name)
        self.dense_model = SentenceTransformer(dense_model_name)
        self.dense_dimension = int(self.dense_model.get_sentence_embedding_dimension())

        logger.info("Đang nạp sparse model BM25: %s", sparse_model_name)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

    @staticmethod
    def clean_query(query: Any) -> str:
        """Xóa HTML/URL, dấu câu và chuẩn hóa khoảng trắng trong query.

        Không stemming, lemmatization hay gọi LLM ở canonical path. Việc query
        không phải tiếng Anh được xem là experimental vì model v1 là English-only.
        """

        if not isinstance(query, str):
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", query)
        cleaned = re.sub(r"(?:https?://|www\.)\S+", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
        return re.sub(r"\s+", " ", cleaned).strip().lower()

    def encode(self, raw_query: str) -> tuple[str, list[float], tuple[list[int], list[float]]]:
        """Làm sạch query rồi tạo dense vector và sparse BM25 vector."""

        clean_query = self.clean_query(raw_query)
        if not clean_query:
            raise ValueError("Truy vấn tìm kiếm không được để trống hoặc chỉ chứa ký tự đặc biệt.")

        dense_vector = self.dense_model.encode(
            clean_query,
            normalize_embeddings=True,
        ).tolist()
        sparse_result = next(iter(self.sparse_model.embed([clean_query])))
        sparse_vector = (sparse_result.indices.tolist(), sparse_result.values.tolist())
        return clean_query, dense_vector, sparse_vector
