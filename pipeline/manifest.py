"""Manifest provenance cho raw/clean dataset snapshots."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval.manifest import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _portable_path(path: Path) -> str:
    """Ghi đường dẫn tương đối để manifest không gắn với máy người chạy."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def dataset_manifest_file(dataset_version: str, output_path: Path | None = None) -> Path:
    """Trả đường dẫn manifest của một raw dataset snapshot."""

    if output_path is not None:
        return Path(output_path)
    return PROJECT_ROOT / "data" / "manifests" / f"{dataset_version}.json"


def write_dataset_manifest(
    dataset_path: Path,
    *,
    dataset_version: str,
    start_year: int,
    end_year: int,
    filters: dict[str, Any],
    movie_count: int | None = None,
    output_path: Path | None = None,
) -> Path:
    """Ghi identity và hash của snapshot TMDB để tái lập ingestion."""

    dataset_path = Path(dataset_path)
    target = dataset_manifest_file(dataset_version, output_path)
    manifest = {
        "dataset_version": dataset_version,
        "source": "TMDB",
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "start_year": start_year,
        "end_year": end_year,
        "filters": filters,
        "movie_count": movie_count,
        "raw_sha256": sha256_file(dataset_path),
        "path": _portable_path(dataset_path),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def update_dataset_manifest(
    dataset_version: str,
    clean_path: Path,
    *,
    clean_movie_count: int,
    document_schema_version: str,
    output_path: Path | None = None,
) -> Path:
    """Bổ sung hash/schema của canonical table vào manifest raw tương ứng."""

    target = dataset_manifest_file(dataset_version, output_path)
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy raw dataset manifest: {target}")

    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset manifest không phải JSON hợp lệ: {target}") from exc
    if manifest.get("dataset_version") != dataset_version:
        raise ValueError("dataset_version trong manifest không khớp canonical table.")

    manifest.update(
        {
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
            "clean_path": _portable_path(Path(clean_path)),
            "clean_sha256": sha256_file(Path(clean_path)),
            "clean_movie_count": clean_movie_count,
            "document_schema": document_schema_version,
        }
    )
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
