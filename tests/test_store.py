"""Test store module và hybrid search parallel execution."""

from unittest.mock import MagicMock, patch

import pytest

from retrieval.store import check_collection_exists, hybrid_search


@patch("retrieval.store.sparse_search")
@patch("retrieval.store.dense_search")
@patch("retrieval.store.check_collection_exists")
def test_hybrid_search_runs_both_branches(mock_check, mock_dense, mock_sparse):
    mock_dense.return_value = [{"id": "d1", "score": 0.8}]
    mock_sparse.return_value = [{"id": "s1", "score": 4.5}]

    dense, sparse = hybrid_search([0.1], ([1], [1.0]), limit=1)

    assert dense == [{"id": "d1", "score": 0.8}]
    assert sparse == [{"id": "s1", "score": 4.5}]
    mock_check.assert_called_once()


def test_check_collection_exists_raises_when_missing():
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False

    with pytest.raises(RuntimeError, match="không tồn tại"):
        check_collection_exists(client=mock_client)
