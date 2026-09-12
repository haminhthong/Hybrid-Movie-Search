"""Build hybrid Qdrant index (Dense MiniLM + Sparse BM25) cho movies."""

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from retrieval.config import settings

from .clean import parse_list_value

logger = logging.getLogger(__name__)

INPUT_FILE = Path(__file__).resolve().parent / "data" / "movies_clean.csv"
BATCH_SIZE = 32
MAX_RETRIES = 3


def normalize_document(value: Any) -> str:
    """Chuẩn hóa whitespace của search document trước khi encode."""
    return " ".join(str(value or "").lower().split())


def safe_int(value: Any, default: int = 0) -> int:
    """Đọc số nguyên từ CSV mà không làm hỏng cả batch."""
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else int(numeric)


def safe_float(value: Any, default: float = 0.0) -> float:
    """Đọc số thực từ CSV mà không để NaN vào payload."""
    numeric = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(numeric) else float(numeric)


def safe_text(value: Any) -> str:
    """Chuyển scalar CSV sang text sạch."""
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return ""
    return str(value).strip()


def movie_point_id(movie_id: int) -> str:
    """Tạo UUID ổn định để cùng movie_id luôn là cùng một Qdrant point."""
    if movie_id <= 0:
        raise ValueError("movie_id phải lớn hơn 0.")
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"movie_{movie_id}"))


def prepare_collection(client: QdrantClient, collection_name: str, dense_size: int) -> None:
    """Tạo mới hoặc tái tạo collection với cấu hình dense và sparse vector."""
    if client.collection_exists(collection_name):
        logger.info("Collection %s đã tồn tại, xóa để tạo mới...", collection_name)
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    # Keyword và integer indexes cho structured filtering
    indexes = {
        "movie_id": models.PayloadSchemaType.INTEGER,
        "genre_keys": models.PayloadSchemaType.KEYWORD,
        "release_year": models.PayloadSchemaType.INTEGER,
    }
    for field_name, field_schema in indexes.items():
        client.create_payload_index(collection_name, field_name, field_schema)


def process_dual_embedding(input_file: Path = INPUT_FILE) -> int:
    """Batch encode dense + sparse và index vào Qdrant."""
    settings.require_qdrant()
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu sạch: {input_path}")

    dataframe = pd.read_csv(input_path)
    required = {"movie_id", "combined_text"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {', '.join(sorted(missing))}")

    dataframe["movie_id"] = pd.to_numeric(dataframe["movie_id"], errors="coerce")
    dataframe = dataframe.dropna(subset=["movie_id"]).copy()
    dataframe["movie_id"] = dataframe["movie_id"].astype(int)
    dataframe = dataframe[(dataframe["movie_id"] > 0) & dataframe["combined_text"].notna()]
    dataframe = dataframe.drop_duplicates(subset=["movie_id"], keep="first")
    dataframe = dataframe[dataframe["combined_text"].map(normalize_document).str.len() > 0]
    if dataframe.empty:
        raise ValueError("Không có movie document hợp lệ để tạo index.")

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60.0)
    dense_model = SentenceTransformer(settings.dense_model)
    dense_size = int(dense_model.get_sentence_embedding_dimension())
    sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)

    collection_name = settings.collection_name
    prepare_collection(client, collection_name, dense_size)

    valid_point_count = 0
    for offset in tqdm(range(0, len(dataframe), BATCH_SIZE), desc="Đang index dữ liệu phim"):
        batch = dataframe.iloc[offset : offset + BATCH_SIZE]
        documents = [normalize_document(value) for value in batch["combined_text"]]
        dense_vectors = dense_model.encode(
            documents,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        sparse_vectors = list(sparse_model.embed(documents))

        points = []
        for position, (_, row) in enumerate(batch.iterrows()):
            movie_id = safe_int(row["movie_id"])
            if movie_id <= 0:
                continue
            genres = parse_list_value(row.get("genres", ""))
            payload = {
                "movie_id": movie_id,
                "title": safe_text(row.get("title", "")),
                "director": safe_text(row.get("director", "")),
                "cast": parse_list_value(row.get("cast", "")),
                "genres": genres,
                "genre_keys": [g.casefold() for g in genres],
                "keywords": parse_list_value(row.get("keywords", "")),
                "overview": safe_text(row.get("overview", "")),
                "document_text": safe_text(row["combined_text"]),
                "release_date": safe_text(row.get("release_date", "")),
                "release_year": safe_int(row.get("release_year", 0)),
                "vote_average": safe_float(row.get("vote_average", 0)),
                "popularity": safe_float(row.get("popularity", 0)),
                "poster_path": safe_text(row.get("poster_path", "")),
            }
            sparse = sparse_vectors[position]
            points.append(
                models.PointStruct(
                    id=movie_point_id(movie_id),
                    vector={
                        "dense": dense_vectors[position],
                        "sparse": models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload=payload,
                )
            )

        if not points:
            continue
        valid_point_count += len(points)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client.upsert(collection_name=collection_name, points=points, wait=True)
                break
            except Exception:
                if attempt == MAX_RETRIES:
                    raise
                logger.exception("Upsert lần %d/%d lỗi; thử lại", attempt, MAX_RETRIES)
                time.sleep(2**attempt)

    actual_count = int(client.count(collection_name=collection_name, exact=True).count)
    logger.info(
        "Index thành công: collection=%s, indexed_points=%d, total_in_qdrant=%d",
        collection_name,
        valid_point_count,
        actual_count,
    )
    return actual_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    process_dual_embedding()
