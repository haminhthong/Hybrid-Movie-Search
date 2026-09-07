"""Đọc và kiểm tra manifest của dataset/index.

Manifest là nguồn sự thật để phát hiện việc query model và index model không
cùng semantic space. Không dựa vào tên collection hoặc chỉ số chiều vector.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def manifest_directory() -> Path:
    """Trả về thư mục manifest, cho phép override bằng biến môi trường."""

    path = Path(settings.manifest_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def manifest_file(index_version: str | None = None) -> Path:
    """Xác định đường dẫn manifest theo version index."""

    return manifest_directory() / f"{index_version or settings.index_version}.json"


def sha256_file(path: Path) -> str:
    """Tính SHA-256 theo luồng để không nạp toàn bộ dataset vào RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    collection_name: str,
    point_count: int,
    dataset_version: str,
    dataset_sha256: str,
    dense_dimension: int,
    index_version: str | None = None,
    source: str = "TMDB",
) -> dict[str, Any]:
    """Tạo manifest đầy đủ cho một collection đã build xong."""

    version = index_version or settings.index_version
    return {
        "index_version": version,
        "collection": collection_name,
        "alias": settings.index_alias,
        "source": source,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_sha256,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dense_model": settings.dense_model,
        "dense_dimension": dense_dimension,
        "sparse_model": settings.sparse_model,
        "document_schema": settings.document_schema_version,
        "point_count": point_count,
    }


def write_manifest(manifest: dict[str, Any], path: Path | None = None) -> Path:
    """Ghi manifest dạng JSON với format ổn định, dễ review trong git."""

    output = Path(path) if path else manifest_file(str(manifest["index_version"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Đọc manifest; ném lỗi rõ ràng nếu artifact chưa tồn tại hoặc JSON hỏng."""

    source = Path(path) if path else manifest_file()
    if not source.exists():
        raise RuntimeError(f"Không tìm thấy index manifest: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Index manifest không phải JSON hợp lệ: {source}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Index manifest phải là một object JSON: {source}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Kiểm tra model/index contract trước khi cho phép query."""

    required = {
        "index_version",
        "collection",
        "alias",
        "dense_model",
        "dense_dimension",
        "sparse_model",
        "document_schema",
        "point_count",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise RuntimeError(f"Index manifest thiếu trường: {', '.join(missing)}")

    expected = {
        "index_version": settings.index_version,
        "alias": settings.index_alias,
        "dense_model": settings.dense_model,
        "dense_dimension": settings.dense_dimension,
        "sparse_model": settings.sparse_model,
        "document_schema": settings.document_schema_version,
    }
    mismatches = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = "; ".join(
            f"{key}: expected={expected_value!r}, actual={actual!r}"
            for key, (expected_value, actual) in mismatches.items()
        )
        raise RuntimeError(f"Index/model contract không tương thích: {details}")
    try:
        point_count = int(manifest["point_count"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Index manifest có point_count không hợp lệ.") from exc
    if point_count <= 0:
        raise RuntimeError("Index manifest có point_count không hợp lệ.")
