"""Tests for /api/products endpoints."""
import pytest


# ── POST /api/products/ ───────────────────────────────────────────────────────

def test_add_product_success(client, mock_clip, mock_pinecone_svc):
    resp = client.post(
        "/api/products/",
        json={
            "id": "shoe-001",
            "name": "Air Max 2024",
            "category": "footwear",
            "price": 120.0,
            "image_url": "http://example.com/airmax.jpg",
            "description": "Nike Air Max",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == "shoe-001"
    assert data["message"] == "Product indexed successfully"


def test_add_product_calls_clip_and_pinecone(client, mock_clip, mock_pinecone_svc):
    client.post(
        "/api/products/",
        json={
            "id": "bag-01",
            "name": "Canvas Tote",
            "category": "bags",
            "price": 45.0,
            "image_url": "http://example.com/bag.jpg",
            "description": "Eco-friendly",
        },
    )
    mock_clip.embed_image_url.assert_called_once()
    mock_pinecone_svc.upsert_product.assert_called_once()


def test_add_product_upserts_correct_metadata(client, mock_clip, mock_pinecone_svc):
    client.post(
        "/api/products/",
        json={
            "id": "p-2",
            "name": "Canvas Bag",
            "category": "bags",
            "price": 45.0,
            "image_url": "http://example.com/bag.jpg",
            "description": "Eco-friendly",
        },
    )
    product_id, embedding, metadata = mock_pinecone_svc.upsert_product.call_args.args
    assert product_id == "p-2"
    assert metadata["name"] == "Canvas Bag"
    assert metadata["category"] == "bags"
    assert metadata["price"] == 45.0
    assert metadata["description"] == "Eco-friendly"


def test_add_product_without_price_stores_none(client, mock_clip, mock_pinecone_svc):
    client.post(
        "/api/products/",
        json={
            "id": "p-3",
            "name": "Mystery Item",
            "category": "other",
            "image_url": "http://example.com/item.jpg",
            "description": "",
        },
    )
    _, _, metadata = mock_pinecone_svc.upsert_product.call_args.args
    assert metadata["price"] is None


def test_add_product_image_error_returns_400(client, mock_clip, mock_pinecone_svc):
    mock_clip.embed_image_url.side_effect = Exception("Image not found")
    resp = client.post(
        "/api/products/",
        json={
            "id": "p-broken",
            "name": "Broken",
            "category": "other",
            "image_url": "http://example.com/broken.jpg",
            "description": "",
        },
    )
    assert resp.status_code == 400
    assert "Could not embed image" in resp.json()["detail"]


def test_add_product_missing_id_returns_422(client, mock_clip, mock_pinecone_svc):
    resp = client.post(
        "/api/products/",
        json={
            "name": "Shoe",
            "category": "footwear",
            "image_url": "http://example.com/shoe.jpg",
        },
    )
    assert resp.status_code == 422


def test_add_product_invalid_image_url_returns_422(client, mock_clip, mock_pinecone_svc):
    resp = client.post(
        "/api/products/",
        json={
            "id": "p-bad",
            "name": "Shoe",
            "category": "footwear",
            "image_url": "not-a-url",
            "description": "",
        },
    )
    assert resp.status_code == 422


# ── DELETE /api/products/{product_id} ────────────────────────────────────────

def test_delete_product_success(client, mock_pinecone_svc):
    resp = client.delete("/api/products/shoe-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "shoe-001"
    assert "removed" in data["message"].lower()
    mock_pinecone_svc.delete_product.assert_called_once_with("shoe-001")


# ── GET /api/products/stats ───────────────────────────────────────────────────

def test_index_stats_returns_pinecone_stats(client, mock_pinecone_svc):
    mock_pinecone_svc.get_stats.return_value = {"total_vector_count": 5000}
    resp = client.get("/api/products/stats")
    assert resp.status_code == 200
    assert resp.json() == {"total_vector_count": 5000}
