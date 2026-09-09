"""Cấu hình dùng chung cho indexing, retrieval và API.

Các giá trị ở đây là một phần của hợp đồng model--index. Khi đổi model hoặc
schema tài liệu, cần tăng ``INDEX_VERSION`` và build index mới trước khi release.
"""

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Alias bất biến mà API sử dụng để đọc index đang phục vụ traffic.
INDEX_ALIAS: str = os.getenv("QDRANT_INDEX_ALIAS", "movies_current").strip() or "movies_current"

# Các model dùng trong production v1. HyDE không còn nằm trong online path.
DENSE_MODEL: str = os.getenv("DENSE_MODEL", "all-MiniLM-L6-v2").strip()
SPARSE_MODEL: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25").strip()
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
# Tăng version khi thay đổi field, thứ tự field hoặc quy tắc chuẩn hóa document.
DOCUMENT_SCHEMA_VERSION: str = "movie-doc-v2"


def _safe_version(value: str) -> str:
    """Giữ tên version an toàn khi dùng làm tên collection Qdrant."""

    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-_.") or "unversioned"


@dataclass(frozen=True)
class Settings:
    """Thông số runtime và contract của index đang được phục vụ."""

    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    index_alias: str = INDEX_ALIAS
    # Không gắn ngày hiện tại vào default để clone mới không giả danh một artifact cũ.
    index_version: str = _safe_version(os.getenv("INDEX_VERSION", "tmdb-local-minilm-v2"))
    dense_model: str = DENSE_MODEL
    sparse_model: str = SPARSE_MODEL
    rerank_model: str = RERANK_MODEL
    dense_dimension: int = int(os.getenv("DENSE_DIMENSION", "384"))
    document_schema_version: str = DOCUMENT_SCHEMA_VERSION
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "50"))
    candidate_k: int = int(os.getenv("FUSION_CANDIDATE_K", "30"))
    rerank_k: int = int(os.getenv("RERANK_K", "20"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    cache_size: int = int(os.getenv("SEARCH_CACHE_SIZE", "128"))
    manifest_path: str = os.getenv("INDEX_MANIFEST_PATH", "data/manifests")

    @property
    def collection_name(self) -> str:
        """Tên collection versioned dùng làm shadow index."""

        return f"movies_{self.index_version}"

    def require_qdrant(self) -> None:
        """Dừng sớm nếu thiếu URL kết nối Qdrant."""

        if not self.qdrant_url:
            raise RuntimeError("Thiếu biến môi trường QDRANT_URL. Vui lòng cấu hình trong tệp .env.")

    def validate(self) -> None:
        """Kiểm tra các tham số retrieval để tránh chạy với cấu hình vô hiệu."""

        if self.dense_dimension <= 0:
            raise ValueError("DENSE_DIMENSION phải lớn hơn 0.")
        if not 0 < self.rerank_k <= self.candidate_k <= self.retrieval_k:
            raise ValueError("Cần thỏa mãn 0 < rerank_k <= candidate_k <= retrieval_k.")
        if self.rrf_k < 0:
            raise ValueError("RRF_K không được âm.")
        if self.cache_size <= 0:
            raise ValueError("SEARCH_CACHE_SIZE phải lớn hơn 0.")


settings = Settings()
settings.validate()

# Tên cũ được giữ như alias để các module bên ngoài không bị vỡ ngay lập tức.
COLLECTION_NAME: str = INDEX_ALIAS
