"""Cấu hình dùng chung cho indexing, retrieval và API."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DENSE_MODEL: str = os.getenv("DENSE_MODEL", "all-MiniLM-L6-v2").strip()
SPARSE_MODEL: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25").strip()
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "movies").strip() or "movies"


@dataclass(frozen=True)
class Settings:
    """Thông số runtime cho hệ thống hybrid retrieval."""

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    collection_name: str = COLLECTION_NAME
    dense_model: str = DENSE_MODEL
    sparse_model: str = SPARSE_MODEL
    rerank_model: str = RERANK_MODEL
    dense_dimension: int = int(os.getenv("DENSE_DIMENSION", "384"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "50"))
    rerank_k: int = int(os.getenv("RERANK_K", "20"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    def require_qdrant(self) -> None:
        """Dừng sớm nếu thiếu URL kết nối Qdrant."""
        if not self.qdrant_url:
            raise RuntimeError("Thiếu biến môi trường QDRANT_URL. Vui lòng cấu hình trong tệp .env.")

    def validate(self) -> None:
        """Kiểm tra các tham số retrieval để tránh chạy với cấu hình không hợp lệ."""
        if self.dense_dimension <= 0:
            raise ValueError("DENSE_DIMENSION phải lớn hơn 0.")
        if self.retrieval_k <= 0 or self.rerank_k <= 0:
            raise ValueError("retrieval_k và rerank_k phải lớn hơn 0.")
        if self.rerank_k > self.retrieval_k:
            raise ValueError("rerank_k không được lớn hơn retrieval_k.")
        if self.rrf_k < 0:
            raise ValueError("RRF_K không được âm.")


settings = Settings()
settings.validate()
