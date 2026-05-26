"""
Shared test configuration.

Patches transformers and pinecone before any app module is imported so
the module-level singletons initialize with lightweight mocks instead of
downloading CLIP weights or connecting to Pinecone.
"""
import os
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

# Ensure backend/ is on sys.path so `from app.xxx import ...` works regardless
# of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Required by pydantic-settings before config.py is parsed at import time.
os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")

# ── Build mock objects ────────────────────────────────────────────────────────

_mock_clip_model = MagicMock()
# .to(device) should return the same mock so clip_service.model is accessible.
_mock_clip_model.to.return_value = _mock_clip_model

_mock_clip_processor = MagicMock()

_mock_pinecone_class = MagicMock()
_mock_pinecone_instance = MagicMock()
# Make the index appear to already exist so _ensure_index skips creation.
_mock_index_entry = MagicMock()
_mock_index_entry.name = "product-search"
_mock_pinecone_instance.indexes.list.return_value = [_mock_index_entry]
_mock_pinecone_instance.index.return_value = MagicMock()
_mock_pinecone_class.return_value = _mock_pinecone_instance

# ── Import app under patches ──────────────────────────────────────────────────
# These patches must be active while app modules are first imported so the
# singleton __init__ calls run against the mocks.

with \
    patch("transformers.CLIPModel.from_pretrained", return_value=_mock_clip_model), \
    patch("transformers.CLIPProcessor.from_pretrained", return_value=_mock_clip_processor), \
    patch("pinecone.Pinecone", _mock_pinecone_class):

    from app.main import app  # triggers all singleton initializations
    from app.services.clip_service import clip_service, CLIPEmbeddingService
    from app.services.pinecone_service import pinecone_service, PineconeService

from fastapi.testclient import TestClient

# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_clip(monkeypatch):
    """Replace clip_service in both router modules with a fresh MagicMock."""
    mock = MagicMock()
    mock.embed_text.return_value = [0.1] * 512
    mock.embed_image_bytes.return_value = [0.1] * 512
    mock.embed_image_url.return_value = [0.1] * 512
    monkeypatch.setattr("app.api.search.clip_service", mock)
    monkeypatch.setattr("app.api.products.clip_service", mock)
    return mock


@pytest.fixture
def mock_pinecone_svc(monkeypatch):
    """Replace pinecone_service in both router modules with a fresh MagicMock."""
    mock = MagicMock()
    mock.query.return_value = []
    monkeypatch.setattr("app.api.search.pinecone_service", mock)
    monkeypatch.setattr("app.api.products.pinecone_service", mock)
    return mock


@pytest.fixture
def sample_result():
    return {
        "id": "prod-001",
        "score": 0.87,
        "name": "Red Running Shoes",
        "category": "footwear",
        "price": 79.99,
        "image_url": "http://example.com/shoe.jpg",
        "description": "Lightweight and fast.",
    }


@pytest.fixture
def jpeg_bytes():
    img = Image.new("RGB", (64, 64), color=(200, 100, 50))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    img = Image.new("RGB", (64, 64), color=(50, 100, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
