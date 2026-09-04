"""Tests for FastAPI HTTP endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_healthcheck():
    """Test GET /health status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_warm_user_two_tower_routing():
    """Test POST /api/recommend with history_count > 10 routes to two_tower."""
    payload = {
        "user_id": 10,
        "history": [101, 204, 305, 400, 500, 600, 700, 800, 900, 1000, 1100]  # 11 items
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == 10
    assert data["strategy_used"] == "two_tower"
    assert len(data["recommendations"]) == 10
    assert len(set(data["recommendations"])) == 10
    assert all(isinstance(x, int) for x in data["recommendations"])


def test_recommend_boundary_history_length_10():
    """Test POST /api/recommend with history_count == 10 routes to content_fallback."""
    payload = {
        "user_id": 10,
        "history": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]  # exactly 10 items
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == 10
    assert data["strategy_used"] == "content_fallback"
    assert len(data["recommendations"]) == 10
    assert len(set(data["recommendations"])) == 10


def test_recommend_history_length_1():
    """Test POST /api/recommend with history_count == 1 routes to content_fallback."""
    payload = {
        "user_id": 10,
        "history": [101]
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == 10
    assert data["strategy_used"] == "content_fallback"
    assert len(data["recommendations"]) == 10
    assert len(set(data["recommendations"])) == 10


def test_recommend_empty_history_popularity_fallback():
    """Test POST /api/recommend with empty history routes to popularity_fallback."""
    payload = {
        "user_id": 10,
        "history": []
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == 10
    assert data["strategy_used"] == "popularity_fallback"
    assert len(data["recommendations"]) == 10
    assert len(set(data["recommendations"])) == 10


def test_recommend_unknown_user_non_empty_history():
    """Test unknown user with non-empty history routes safely to content_fallback."""
    payload = {
        "user_id": 99999,
        "history": [101]
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == 99999
    assert data["strategy_used"] == "content_fallback"
    assert len(data["recommendations"]) == 10
    assert len(set(data["recommendations"])) == 10
    assert all(isinstance(x, int) for x in data["recommendations"])


def test_recommend_unknown_user_empty_history():
    """Test unknown user with empty history routes safely to popularity_fallback without embedding errors."""
    payload = {
        "user_id": 99999,
        "history": []
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == 99999
    assert data["strategy_used"] == "popularity_fallback"
    assert len(data["recommendations"]) == 10
    assert len(set(data["recommendations"])) == 10
    assert all(isinstance(x, int) for x in data["recommendations"])


def test_recommend_invalid_payload_type():
    """Test invalid request payload returns HTTP 422 validation error."""
    payload = {
        "user_id": "not_an_int",
        "history": [101]
    }
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 422

