"""Test FastAPI endpoints và response validation."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_endpoint_validation_error():
    response = client.post("/search", json={"query": "", "top_n": 10})
    assert response.status_code == 422


def test_search_endpoint_success_mocked():
    mock_engine = MagicMock()
    mock_engine.search.return_value = {
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
            }
        ],
        "latency_ms": 42.5,
    }
    with patch("app.api.get_search_engine", return_value=mock_engine):
        response = client.post(
            "/search",
            json={"query": "astronauts traveling through wormhole", "top_n": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["title"] == "Interstellar"
        assert data["results"][0]["genres"] == ["Adventure", "Drama", "Science Fiction"]
        assert data["latency_ms"] == 42.5
