"""Manifest provenance cho raw/clean dataset snapshots."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retrieval.manifest import sha256_file


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
    target = output_path or (
        dataset_path.parent.parent.parent / "data" / "manifests" / f"{dataset_version}.json"
    )
    manifest = {
        "dataset_version": dataset_version,
        "source": "TMDB",
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "start_year": start_year,
        "end_year": end_year,
        "filters": filters,
        "movie_count": movie_count,
        "raw_sha256": sha256_file(dataset_path),
        "path": str(dataset_path),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
