"""Test FastAPI liveness, validation và search response contract."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api import app, get_service

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_endpoint_validation_error():
    response = client.post("/search", json={"query": "", "top_n": 10})
    assert response.status_code == 422


def test_search_endpoint_success_mocked():
    mock_service = MagicMock()
    mock_service.search.return_value = {
        "query": "astronauts traveling through wormhole",
        "results": [
            {
                "movie_id": 157336,
                "title": "Interstellar",
                "director": "Christopher Nolan",
                "cast": ["Matthew McConaughey"],
                "genres": ["Adventure", "Drama", "Science Fiction"],
                "keywords": ["space"],
                "overview": "A team of explorers travels through a wormhole.",
                "release_date": "2014-11-05",
                "release_year": 2014,
                "vote_average": 8.4,
                "popularity": 145.2,
                "poster_path": "/poster.jpg",
                "rank": 1,
                "rerank_score": 6.42,
                "display_score": 1.0,
            }
        ],
        "index_version": "tmdb-20260907-minilm-v1",
        "latency_ms": 42.5,
    }
    app.dependency_overrides[get_service] = lambda: mock_service
    try:
        response = client.post(
            "/search",
            json={"query": "astronauts traveling through wormhole", "top_n": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["title"] == "Interstellar"
        assert data["results"][0]["genres"] == ["Adventure", "Drama", "Science Fiction"]
        assert data["index_version"] == "tmdb-20260907-minilm-v1"
    finally:
        app.dependency_overrides.clear()
