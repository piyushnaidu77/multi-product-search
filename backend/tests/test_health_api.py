"""Tests for GET /api/health."""


def test_health_returns_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_method_not_allowed(client):
    resp = client.post("/api/health")
    assert resp.status_code == 405
