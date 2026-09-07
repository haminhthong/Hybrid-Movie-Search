"""Test model--index manifest contract."""

import pytest

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
