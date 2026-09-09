"""Test orchestration canonical và metadata filters."""

from unittest.mock import MagicMock, patch

from retrieval.query import QueryEncoder
from retrieval.search import MovieSearch, build_filter


def test_build_filter_uses_case_insensitive_exact_genre_match():
    q_filter = build_filter(genre="Action")
    assert q_filter is not None
    assert len(q_filter.must) == 1
    assert q_filter.must[0].key == "genre_keys"
    assert q_filter.must[0].match.value == "action"


def test_build_filter_year_only():
    q_filter = build_filter(year="2014")
    assert q_filter is not None
    assert len(q_filter.must) == 1


def test_build_filter_empty():
    assert build_filter(genre="All", year="") is None
    assert build_filter(genre="", year="") is None


def test_query_normalization_is_lightweight_and_deterministic():
    assert QueryEncoder.clean_query(" <b>Interstellar</b>  https://example.com ") == "interstellar"


@patch("retrieval.search.hybrid_search")
def test_movie_search_runs_rrf_then_cross_encoder(mock_hybrid_search):
    encoder = MagicMock()
    encoder.encode.return_value = ("clean query", [0.1] * 384, ([1], [1.0]))
    reranker = MagicMock()

    mock_hybrid_search.return_value = (
        [{"id": "doc1", "score": 0.9, "payload": {"movie_id": 1, "title": "Interstellar"}}],
        [{"id": "doc1", "score": 10.0, "payload": {"movie_id": 1, "title": "Interstellar"}}],
    )

    def rerank(query, movies):
        movies[0]["rerank_score"] = 4.2
        return movies

    reranker.rerank.side_effect = rerank
    response = MovieSearch(encoder=encoder, reranker=reranker).search("space wormhole", top_n=5)

    assert response["query"] == "clean query"
    assert response["results"][0]["title"] == "Interstellar"
    assert response["results"][0]["rank"] == 1
    assert response["results"][0]["rerank_score"] == 4.2
    assert "route" not in response
    assert "hyde" not in response
    reranker.rerank.assert_called_once()
