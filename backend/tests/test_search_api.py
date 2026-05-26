"""Tests for /api/search endpoints and search helper functions."""
import pytest

from app.api.search import _build_filter, _format_response


# ── _build_filter ─────────────────────────────────────────────────────────────

def test_build_filter_with_category():
    assert _build_filter("shoes") == {"category": {"$eq": "shoes"}}


def test_build_filter_returns_none_when_no_category():
    assert _build_filter(None) is None


def test_build_filter_returns_none_for_empty_string():
    # Empty string is falsy; no filter should be applied.
    assert _build_filter("") is None


# ── _format_response ──────────────────────────────────────────────────────────

def test_format_response_populates_fields(sample_result):
    resp = _format_response([sample_result], "text")
    assert resp.query_type == "text"
    assert resp.total == 1
    assert resp.results[0].id == "prod-001"
    assert resp.results[0].score == 0.87
    assert resp.results[0].name == "Red Running Shoes"


def test_format_response_empty_list():
    resp = _format_response([], "image")
    assert resp.total == 0
    assert resp.results == []


def test_format_response_total_matches_result_count(sample_result):
    resp = _format_response([sample_result, sample_result], "image_url")
    assert resp.total == len(resp.results) == 2


# ── POST /api/search/text ─────────────────────────────────────────────────────

def test_text_search_success(client, mock_clip, mock_pinecone_svc, sample_result):
    mock_pinecone_svc.query.return_value = [sample_result]
    resp = client.post("/api/search/text", json={"query": "red shoes"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_type"] == "text"
    assert data["total"] == 1
    assert data["results"][0]["id"] == "prod-001"
    mock_clip.embed_text.assert_called_once_with("red shoes")


def test_text_search_empty_query_returns_400(client, mock_clip, mock_pinecone_svc):
    resp = client.post("/api/search/text", json={"query": ""})
    assert resp.status_code == 400


def test_text_search_whitespace_only_returns_400(client, mock_clip, mock_pinecone_svc):
    resp = client.post("/api/search/text", json={"query": "   "})
    assert resp.status_code == 400


def test_text_search_with_category_passes_filter(client, mock_clip, mock_pinecone_svc):
    mock_pinecone_svc.query.return_value = []
    client.post("/api/search/text", json={"query": "boots", "category": "footwear"})
    kwargs = mock_pinecone_svc.query.call_args.kwargs
    assert kwargs["filter"] == {"category": {"$eq": "footwear"}}


def test_text_search_without_category_passes_none_filter(client, mock_clip, mock_pinecone_svc):
    mock_pinecone_svc.query.return_value = []
    client.post("/api/search/text", json={"query": "boots"})
    kwargs = mock_pinecone_svc.query.call_args.kwargs
    assert kwargs["filter"] is None


def test_text_search_respects_top_k(client, mock_clip, mock_pinecone_svc):
    mock_pinecone_svc.query.return_value = []
    client.post("/api/search/text", json={"query": "hat", "top_k": 5})
    kwargs = mock_pinecone_svc.query.call_args.kwargs
    assert kwargs["top_k"] == 5


def test_text_search_missing_query_field_returns_422(client, mock_clip, mock_pinecone_svc):
    resp = client.post("/api/search/text", json={})
    assert resp.status_code == 422


# ── POST /api/search/image ────────────────────────────────────────────────────

def test_image_search_jpeg_success(client, mock_clip, mock_pinecone_svc, jpeg_bytes, sample_result):
    mock_pinecone_svc.query.return_value = [sample_result]
    resp = client.post(
        "/api/search/image",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_type"] == "image"
    assert data["total"] == 1
    mock_clip.embed_image_bytes.assert_called_once()


def test_image_search_png_success(client, mock_clip, mock_pinecone_svc, png_bytes):
    mock_pinecone_svc.query.return_value = []
    resp = client.post(
        "/api/search/image",
        files={"file": ("photo.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200


def test_image_search_unsupported_content_type_returns_400(client, mock_clip, mock_pinecone_svc):
    resp = client.post(
        "/api/search/image",
        files={"file": ("anim.gif", b"GIF89a\x01", "image/gif")},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


def test_image_search_oversized_file_returns_400(client, mock_clip, mock_pinecone_svc):
    # JPEG magic bytes + padding that pushes the file just over 5 MB.
    over_limit = b"\xff\xd8\xff" + b"x" * (5 * 1024 * 1024 + 1)
    resp = client.post(
        "/api/search/image",
        files={"file": ("big.jpg", over_limit, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "5MB" in resp.json()["detail"]


def test_image_search_with_category_query_param(client, mock_clip, mock_pinecone_svc, jpeg_bytes):
    mock_pinecone_svc.query.return_value = []
    client.post(
        "/api/search/image",
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
        params={"category": "bags"},
    )
    kwargs = mock_pinecone_svc.query.call_args.kwargs
    assert kwargs["filter"] == {"category": {"$eq": "bags"}}


# ── POST /api/search/url ──────────────────────────────────────────────────────

def test_url_search_success(client, mock_clip, mock_pinecone_svc, sample_result):
    mock_pinecone_svc.query.return_value = [sample_result]
    resp = client.post(
        "/api/search/url",
        json={"image_url": "http://example.com/shoe.jpg"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_type"] == "image_url"
    assert data["total"] == 1
    mock_clip.embed_image_url.assert_called_once()


def test_url_search_returns_400_when_image_fetch_fails(client, mock_clip, mock_pinecone_svc):
    mock_clip.embed_image_url.side_effect = Exception("Connection refused")
    resp = client.post(
        "/api/search/url",
        json={"image_url": "http://example.com/missing.jpg"},
    )
    assert resp.status_code == 400
    assert "Could not fetch image" in resp.json()["detail"]


def test_url_search_invalid_url_returns_422(client, mock_clip, mock_pinecone_svc):
    resp = client.post(
        "/api/search/url",
        json={"image_url": "not-a-url"},
    )
    assert resp.status_code == 422
