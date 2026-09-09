"""Test model--index manifest contract."""

import json

import pytest

from pipeline.manifest import update_dataset_manifest, write_dataset_manifest
from retrieval.config import settings
from retrieval.manifest import build_manifest, validate_manifest


def test_manifest_accepts_current_contract():
    manifest = build_manifest(
        collection_name=settings.collection_name,
        point_count=3,
        dataset_version="tmdb-test",
        dataset_sha256="abc",
        dense_dimension=settings.dense_dimension,
    )
    validate_manifest(manifest)


def test_manifest_rejects_wrong_dense_model():
    manifest = build_manifest(
        collection_name=settings.collection_name,
        point_count=3,
        dataset_version="tmdb-test",
        dataset_sha256="abc",
        dense_dimension=settings.dense_dimension,
    )
    manifest["dense_model"] = "another-model"
    with pytest.raises(RuntimeError, match="không tương thích"):
        validate_manifest(manifest)


def test_dataset_manifest_links_raw_and_clean_files(tmp_path):
    raw_path = tmp_path / "movies_raw.csv"
    clean_path = tmp_path / "movies_clean.csv"
    manifest_path = tmp_path / "dataset.json"
    raw_path.write_text("movie_id,title\n1,Example\n", encoding="utf-8")
    clean_path.write_text("movie_id,combined_text\n1,Title: Example\n", encoding="utf-8")

    write_dataset_manifest(
        raw_path,
        dataset_version="tmdb-test",
        start_year=2000,
        end_year=2001,
        filters={"overview_required": True},
        movie_count=1,
        output_path=manifest_path,
    )
    update_dataset_manifest(
        "tmdb-test",
        clean_path,
        clean_movie_count=1,
        document_schema_version=settings.document_schema_version,
        output_path=manifest_path,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == "tmdb-test"
    assert manifest["clean_movie_count"] == 1
    assert manifest["document_schema"] == settings.document_schema_version
    assert len(manifest["raw_sha256"]) == 64
    assert len(manifest["clean_sha256"]) == 64
