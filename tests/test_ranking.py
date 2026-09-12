"""Test RRF fusion, structured payload parsing và movie ranking."""

from retrieval.ranking import rank_movies, reciprocal_rank_fusion, to_movies


def test_reciprocal_rank_fusion_is_deterministic():
    dense = [
        {"id": "doc1", "score": 0.9, "payload": {"movie_id": 1, "title": "Movie A"}},
        {"id": "doc2", "score": 0.8, "payload": {"movie_id": 2, "title": "Movie B"}},
    ]
    sparse = [
        {"id": "doc2", "score": 5.0, "payload": {"movie_id": 2, "title": "Movie B"}},
        {"id": "doc3", "score": 4.0, "payload": {"movie_id": 3, "title": "Movie C"}},
    ]

    fused = reciprocal_rank_fusion(dense, sparse, limit=10, rrf_k=60)
    assert [item["id"] for item in fused] == ["doc2", "doc1", "doc3"]
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["sparse_rank"] == 1


def test_to_movies_keeps_structured_metadata():
    docs = [
        {
            "id": "doc1",
            "score": 0.03,
            "payload": {
                "movie_id": 101,
                "title": "Inception",
                "genres": ["Action", "Science Fiction"],
                "cast": ["Leonardo DiCaprio"],
                "release_year": 2010,
                "document_text": "Overview text...",
            },
        },
        {"id": "doc1_dup", "score": 0.02, "payload": {"movie_id": 101, "title": "Duplicate"}},
    ]
    movies = to_movies(docs)
    assert len(movies) == 1
    assert movies[0]["genres"] == ["Action", "Science Fiction"]
    assert movies[0]["cast"] == ["Leonardo DiCaprio"]


def test_rank_movies_orders_by_score_and_assigns_rank():
    movies = [
        {"movie_id": 1, "rerank_score": 10.0},
        {"movie_id": 2, "rerank_score": 15.0},
        {"movie_id": 3, "rerank_score": 5.0},
    ]
    results = rank_movies(movies, top_n=3)
    assert [movie["movie_id"] for movie in results] == [2, 1, 3]
    assert [movie["rank"] for movie in results] == [1, 2, 3]
