"""Build shadow Qdrant index rồi release qua alias atomic.

Flow: đọc canonical table → batch encode → tạo collection versioned → validate
→ ghi index manifest → switch ``movies_current``. Collection đang phục vụ không
bao giờ bị upsert trực tiếp trong lúc build.
"""

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
from retrieval.manifest import build_manifest, sha256_file, write_manifest

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
    """Tạo UUID ổn định để cùng movie luôn là cùng một Qdrant point."""

    if movie_id <= 0:
        raise ValueError("movie_id phải lớn hơn 0.")
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"movie_{movie_id}"))


def _alias_target(client: QdrantClient, alias_name: str) -> str | None:
    """Tìm collection hiện đang được alias trỏ tới."""

    aliases = client.get_aliases().aliases
    return next((item.collection_name for item in aliases if item.alias_name == alias_name), None)


def _prepare_collection(client: QdrantClient, collection_name: str, dense_size: int) -> None:
    """Tạo collection sạch; chỉ xóa collection shadow không đang serve."""

    if client.collection_exists(collection_name):
        if _alias_target(client, settings.index_alias) == collection_name:
            raise RuntimeError("Không được rebuild collection đang được alias phục vụ traffic.")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=dense_size, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    # Keyword index cho list metadata giúp genre filter là categorical exact-match.
    indexes = {
        "movie_id": models.PayloadSchemaType.INTEGER,
        "title": models.PayloadSchemaType.TEXT,
        "director": models.PayloadSchemaType.KEYWORD,
        "cast": models.PayloadSchemaType.KEYWORD,
        "genres": models.PayloadSchemaType.KEYWORD,
        "keywords": models.PayloadSchemaType.KEYWORD,
        "release_date": models.PayloadSchemaType.TEXT,
        "release_year": models.PayloadSchemaType.INTEGER,
        "document_schema_version": models.PayloadSchemaType.KEYWORD,
    }
    for field_name, field_schema in indexes.items():
        client.create_payload_index(collection_name, field_name, field_schema)


def _validate_collection(
    client: QdrantClient,
    collection_name: str,
    expected_count: int,
    dense_size: int,
    smoke_vector: list[float],
) -> None:
    """Quality gate trước khi release collection."""

    actual_count = int(client.count(collection_name=collection_name, exact=True).count)
    if actual_count != expected_count:
        raise RuntimeError(
            f"Point count không khớp: expected={expected_count}, actual={actual_count}."
        )

    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    dense_config = vectors.get("dense") if isinstance(vectors, dict) else vectors
    actual_size = int(dense_config.size)
    if actual_size != dense_size:
        raise RuntimeError(f"Dense dimension không khớp: expected={dense_size}, actual={actual_size}.")

    smoke = client.query_points(
        collection_name=collection_name,
        using="dense",
        query=smoke_vector,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not smoke.points:
        raise RuntimeError("Smoke query không trả về kết quả.")
    payload = smoke.points[0].payload or {}
    required_payload = {"movie_id", "title", "genres", "cast", "director", "document_schema_version"}
    missing_payload = required_payload.difference(payload)
    if missing_payload:
        raise RuntimeError(
            f"Payload schema thiếu trường: {', '.join(sorted(missing_payload))}."
        )


def _switch_alias(client: QdrantClient, collection_name: str) -> None:
    """Đổi alias trong một request Qdrant duy nhất."""

    operations: list[Any] = []
    current = _alias_target(client, settings.index_alias)
    if current:
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=settings.index_alias))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(
                collection_name=collection_name,
                alias_name=settings.index_alias,
            )
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)


def process_dual_embedding(input_file: Path = INPUT_FILE) -> dict[str, Any]:
    """Build và release một version index mới từ canonical movie table."""

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
    if dense_size != settings.dense_dimension:
        raise RuntimeError(
            "Dense model dimension không khớp settings: "
            f"expected={settings.dense_dimension}, actual={dense_size}."
        )
    sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
    collection_name = settings.collection_name
    _prepare_collection(client, collection_name, dense_size)

    first_dense_vector: list[float] | None = None
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
        if first_dense_vector is None and dense_vectors:
            first_dense_vector = dense_vectors[0]

        points = []
        for position, (_, row) in enumerate(batch.iterrows()):
            movie_id = safe_int(row["movie_id"])
            if movie_id <= 0:
                continue
            payload = {
                "movie_id": movie_id,
                "title": safe_text(row.get("title", "")),
                "director": safe_text(row.get("director", "")),
                "cast": parse_list_value(row.get("cast", "")),
                "genres": parse_list_value(row.get("genres", "")),
                "keywords": parse_list_value(row.get("keywords", "")),
                "overview": safe_text(row.get("overview", "")),
                "document_text": safe_text(row["combined_text"]),
                "release_date": safe_text(row.get("release_date", "")),
                "release_year": safe_int(row.get("release_year", 0)),
                "vote_average": safe_float(row.get("vote_average", 0)),
                "popularity": safe_float(row.get("popularity", 0)),
                "poster_path": safe_text(row.get("poster_path", "")),
                "document_schema_version": settings.document_schema_version,
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

    if first_dense_vector is None:
        raise RuntimeError("Không tạo được dense vector nào.")
    _validate_collection(client, collection_name, valid_point_count, dense_size, first_dense_vector)

    dataset_version = settings.index_version.split("-minilm", 1)[0]
    manifest = build_manifest(
        collection_name=collection_name,
        point_count=valid_point_count,
        dataset_version=dataset_version,
        dataset_sha256=sha256_file(input_path),
        dense_dimension=dense_size,
    )
    manifest_path = write_manifest(manifest)
    _switch_alias(client, collection_name)
    logger.info(
        "Release index thành công: collection=%s alias=%s points=%d manifest=%s",
        collection_name,
        settings.index_alias,
        valid_point_count,
        manifest_path,
    )
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    process_dual_embedding()
