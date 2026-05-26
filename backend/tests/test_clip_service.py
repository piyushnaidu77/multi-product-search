"""
Unit tests for CLIPEmbeddingService.

The CLIP model and processor are MagicMocks (patched in conftest.py).
Tests that exercise embed_* methods configure those mocks to return real
torch tensors so that the _normalize path executes with actual math.
"""
import pytest
import torch
import numpy as np
from io import BytesIO
from unittest.mock import MagicMock, patch

import requests as req_lib
from PIL import Image

from app.services.clip_service import clip_service, CLIPEmbeddingService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _red_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── Fixture: configure model mock to return real tensors ─────────────────────

@pytest.fixture
def tensor_outputs():
    """Make clip_service.model return torch.ones(1, 512) for both modalities."""
    vec = torch.ones(1, 512)
    dummy_out = MagicMock()
    clip_service.model.text_model.return_value = dummy_out
    clip_service.model.text_projection.return_value = vec
    clip_service.model.vision_model.return_value = dummy_out
    clip_service.model.visual_projection.return_value = vec
    return clip_service


# ── _normalize ────────────────────────────────────────────────────────────────

def test_normalize_produces_unit_vector():
    vec = torch.tensor([[3.0, 4.0]])  # L2 norm = 5
    result = clip_service._normalize(vec)
    np.testing.assert_allclose(result.norm().item(), 1.0, atol=1e-6)


def test_normalize_preserves_direction():
    vec = torch.tensor([[1.0, 2.0, 3.0]])
    result = clip_service._normalize(vec)
    orig_unit = vec / vec.norm()
    # Dot product of two unit vectors pointing the same way equals 1.
    np.testing.assert_allclose((result * orig_unit).sum().item(), 1.0, atol=1e-6)


def test_normalize_output_is_float32_on_cpu():
    vec = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    result = clip_service._normalize(vec)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"


# ── embed_text ────────────────────────────────────────────────────────────────

def test_embed_text_returns_512_floats(tensor_outputs):
    result = clip_service.embed_text("red running shoes")
    assert isinstance(result, list)
    assert len(result) == 512
    assert all(isinstance(x, float) for x in result)


def test_embed_text_output_is_normalized(tensor_outputs):
    result = clip_service.embed_text("blue jeans")
    norm = sum(x ** 2 for x in result) ** 0.5
    np.testing.assert_allclose(norm, 1.0, atol=1e-5)


# ── embed_image ───────────────────────────────────────────────────────────────

def test_embed_image_returns_512_floats(tensor_outputs):
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    result = clip_service.embed_image(img)
    assert isinstance(result, list)
    assert len(result) == 512
    assert all(isinstance(x, float) for x in result)


# ── embed_image_bytes ─────────────────────────────────────────────────────────

def test_embed_image_bytes_returns_512_floats(tensor_outputs):
    result = clip_service.embed_image_bytes(_red_jpeg_bytes())
    assert isinstance(result, list)
    assert len(result) == 512


def test_embed_image_bytes_invalid_data_raises():
    with pytest.raises(Exception):
        clip_service.embed_image_bytes(b"definitely-not-an-image")


# ── embed_image_url ───────────────────────────────────────────────────────────

def test_embed_image_url_success(tensor_outputs):
    mock_resp = MagicMock()
    mock_resp.content = _red_jpeg_bytes()
    mock_resp.raise_for_status.return_value = None

    with patch("app.services.clip_service.requests.get", return_value=mock_resp):
        result = clip_service.embed_image_url("http://example.com/shoe.jpg")

    assert isinstance(result, list)
    assert len(result) == 512


def test_embed_image_url_propagates_http_error():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = req_lib.HTTPError("404 Not Found")

    with patch("app.services.clip_service.requests.get", return_value=mock_resp):
        with pytest.raises(req_lib.HTTPError):
            clip_service.embed_image_url("http://example.com/missing.jpg")


def test_embed_image_url_passes_correct_url():
    mock_resp = MagicMock()
    mock_resp.content = _red_jpeg_bytes()
    mock_resp.raise_for_status.return_value = None

    with patch("app.services.clip_service.requests.get", return_value=mock_resp) as mock_get:
        clip_service.embed_image_url("http://example.com/item.jpg")

    mock_get.assert_called_once_with("http://example.com/item.jpg", timeout=10)


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_returns_same_instance():
    a = CLIPEmbeddingService()
    b = CLIPEmbeddingService()
    assert a is b


def test_singleton_is_marked_initialized():
    assert clip_service._initialized is True
