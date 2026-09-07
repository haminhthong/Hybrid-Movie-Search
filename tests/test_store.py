"""Test graceful degradation của hai nhánh retrieval."""

from unittest.mock import patch

import pytest

from retrieval.store import hybrid_search


@patch("retrieval.store.sparse_search")
@patch("retrieval.store.dense_search")
@patch("retrieval.store.validate_index_contract")
def test_hybrid_search_survives_one_failed_branch(mock_validate, mock_dense, mock_sparse):
    mock_dense.side_effect = RuntimeError("dense down")
    mock_sparse.return_value = [{"id": "s1"}]

    dense, sparse = hybrid_search([0.1], ([1], [1.0]), limit=1)

    assert dense == []
    assert sparse == [{"id": "s1"}]


@patch("retrieval.store.sparse_search")
@patch("retrieval.store.dense_search")
@patch("retrieval.store.validate_index_contract")
def test_hybrid_search_fails_when_both_branches_are_down(mock_validate, mock_dense, mock_sparse):
    mock_dense.side_effect = RuntimeError("dense down")
    mock_sparse.side_effect = RuntimeError("sparse down")

    with pytest.raises(RuntimeError, match="cả hai nhánh"):
        hybrid_search([0.1], ([1], [1.0]), limit=1)
