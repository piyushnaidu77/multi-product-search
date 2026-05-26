"""
Unit tests for PineconeService.

The Pinecone client is a MagicMock (patched in conftest.py). Tests interact
with pinecone_service.index (also a MagicMock) and verify that service
methods construct and forward calls correctly.
"""
import pytest
from unittest.mock import MagicMock, call

from app.services.pinecone_service import pinecone_service, VECTOR_DIMENSION
from app.core.config import settings


@pytest.fixture(autouse=True)
def reset_index_mock():
    """Reset the index mock before each test to avoid call-count bleed."""
    pinecone_service.index.reset_mock()
    yield


# ── upsert_product ────────────────────────────────────────────────────────────

def test_upsert_product_calls_index_upsert():
    embedding = [0.1] * 512
    metadata = {
        "name": "Test Shoe",
        "category": "footwear",
        "price": 59.99,
        "image_url": "http://x.com/shoe.jpg",
        "description": "Fast",
    }
    pinecone_service.upsert_product("p-1", embedding, metadata)
    pinecone_service.index.upsert.assert_called_once_with(
        vectors=[{"id": "p-1", "values": embedding, "metadata": metadata}]
    )


# ── upsert_batch ──────────────────────────────────────────────────────────────

def test_upsert_batch_single_batch_when_under_limit():
    vectors = [{"id": f"p{i}", "values": [0.0] * 512, "metadata": {}} for i in range(50)]
    pinecone_service.upsert_batch(vectors)
    assert pinecone_service.index.upsert.call_count == 1


def test_upsert_batch_splits_into_correct_number_of_batches():
    vectors = [{"id": f"p{i}", "values": [0.0] * 512, "metadata": {}} for i in range(250)]
    pinecone_service.upsert_batch(vectors, batch_size=100)
    # 250 items / 100 per batch = 3 batches (100 + 100 + 50)
    assert pinecone_service.index.upsert.call_count == 3


def test_upsert_batch_last_batch_contains_remainder():
    vectors = [{"id": f"p{i}", "values": [0.0] * 512, "metadata": {}} for i in range(130)]
    pinecone_service.upsert_batch(vectors, batch_size=100)
    last_call_vectors = pinecone_service.index.upsert.call_args_list[-1].kwargs["vectors"]
    assert len(last_call_vectors) == 30


def test_upsert_batch_empty_list_makes_no_calls():
    pinecone_service.upsert_batch([])
    pinecone_service.index.upsert.assert_not_called()


# ── query ─────────────────────────────────────────────────────────────────────

def test_query_returns_formatted_result_dicts():
    mock_match = MagicMock()
    mock_match.id = "prod-1"
    mock_match.score = 0.95
    mock_match.metadata = {
        "name": "Sneakers",
        "category": "shoes",
        "price": 89.99,
        "image_url": "http://x.com/s.jpg",
        "description": "Nice",
    }
    pinecone_service.index.query.return_value.matches = [mock_match]

    results = pinecone_service.query([0.1] * 512, top_k=5)

    assert len(results) == 1
    assert results[0] == {
        "id": "prod-1",
        "score": 0.95,
        "name": "Sneakers",
        "category": "shoes",
        "price": 89.99,
        "image_url": "http://x.com/s.jpg",
        "description": "Nice",
    }


def test_query_excludes_results_below_min_score():
    low = MagicMock()
    low.id = "low"
    low.score = settings.MIN_SIMILARITY_SCORE - 0.01
    low.metadata = {"name": "", "category": "", "price": None, "image_url": "", "description": ""}

    high = MagicMock()
    high.id = "high"
    high.score = settings.MIN_SIMILARITY_SCORE + 0.10
    high.metadata = {"name": "", "category": "", "price": None, "image_url": "", "description": ""}

    pinecone_service.index.query.return_value.matches = [high, low]
    results = pinecone_service.query([0.1] * 512)

    ids = [r["id"] for r in results]
    assert "high" in ids
    assert "low" not in ids


def test_query_score_is_rounded_to_4_places():
    m = MagicMock()
    m.id = "p"
    m.score = 0.987654321
    m.metadata = {"name": "", "category": "", "price": None, "image_url": "", "description": ""}
    pinecone_service.index.query.return_value.matches = [m]

    results = pinecone_service.query([0.1] * 512)
    assert results[0]["score"] == round(0.987654321, 4)


def test_query_uses_settings_top_k_when_none_passed():
    pinecone_service.index.query.return_value.matches = []
    pinecone_service.query([0.1] * 512, top_k=None)
    kwargs = pinecone_service.index.query.call_args.kwargs
    assert kwargs["top_k"] == settings.TOP_K_RESULTS


def test_query_passes_explicit_top_k():
    pinecone_service.index.query.return_value.matches = []
    pinecone_service.query([0.1] * 512, top_k=7)
    kwargs = pinecone_service.index.query.call_args.kwargs
    assert kwargs["top_k"] == 7


def test_query_forwards_filter_to_index():
    pinecone_service.index.query.return_value.matches = []
    filt = {"category": {"$eq": "shoes"}}
    pinecone_service.query([0.1] * 512, filter=filt)
    kwargs = pinecone_service.index.query.call_args.kwargs
    assert kwargs["filter"] == filt


def test_query_requests_metadata():
    pinecone_service.index.query.return_value.matches = []
    pinecone_service.query([0.1] * 512)
    kwargs = pinecone_service.index.query.call_args.kwargs
    assert kwargs["include_metadata"] is True


# ── delete_product ────────────────────────────────────────────────────────────

def test_delete_product_calls_index_delete():
    pinecone_service.delete_product("prod-99")
    pinecone_service.index.delete.assert_called_once_with(ids=["prod-99"])


# ── get_stats ─────────────────────────────────────────────────────────────────

def test_get_stats_delegates_to_index():
    mock_stats = {"total_vector_count": 1234}
    pinecone_service.index.describe_index_stats.return_value = mock_stats
    assert pinecone_service.get_stats() == mock_stats


# ── _ensure_index ─────────────────────────────────────────────────────────────

def test_ensure_index_skips_create_when_index_exists():
    existing = MagicMock()
    existing.name = settings.PINECONE_INDEX_NAME
    pinecone_service.pc.indexes.list.return_value = [existing]
    pinecone_service.pc.indexes.create.reset_mock()

    pinecone_service._ensure_index()

    pinecone_service.pc.indexes.create.assert_not_called()


def test_ensure_index_creates_index_when_missing():
    pinecone_service.pc.indexes.list.return_value = []
    pinecone_service.pc.indexes.create.reset_mock()

    pinecone_service._ensure_index()

    pinecone_service.pc.indexes.create.assert_called_once()


def test_ensure_index_creates_with_correct_params():
    pinecone_service.pc.indexes.list.return_value = []
    pinecone_service.pc.indexes.create.reset_mock()

    pinecone_service._ensure_index()

    kwargs = pinecone_service.pc.indexes.create.call_args.kwargs
    assert kwargs["name"] == settings.PINECONE_INDEX_NAME
    assert kwargs["dimension"] == VECTOR_DIMENSION
    assert kwargs["metric"] == "cosine"
