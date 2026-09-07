"""Test cache normalization và invalidation theo index version."""

from unittest.mock import MagicMock

from retrieval.config import settings
from retrieval.service import SearchService


def test_search_cache_uses_normalized_query_and_index_version():
    engine = MagicMock()
    engine.search.return_value = {
        "query": "interstellar",
        "results": [],
        "index_version": settings.index_version,
    }
    service = SearchService(engine=engine)

    service.search(" Interstellar ", top_n=1)
    service.search("interstellar", top_n=1)

    engine.search.assert_called_once_with("interstellar", 1, "", "", False)
    assert list(service.cache)[0][0] == settings.index_version
