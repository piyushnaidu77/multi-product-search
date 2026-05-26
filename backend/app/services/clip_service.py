"""
CLIP Embedding Service
----------------------
Generates 512-dim embeddings for both images and text using
OpenAI's CLIP (ViT-B/32). Embeddings from both modalities live
in the same vector space, enabling cross-modal similarity search.
"""

import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class CLIPEmbeddingService:
    _instance = None

    def __new__(cls):
        """Singleton — load model once, reuse across requests."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Loading CLIP model: {settings.CLIP_MODEL_NAME}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        self.model.eval()
        self._initialized = True
        logger.info(f"CLIP loaded on {self.device}")

    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            features = self.model.text_projection(text_outputs.pooler_output)
        return self._normalize(features).tolist()[0]

    def embed_image(self, image: Image.Image) -> list[float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            features = self.model.visual_projection(vision_outputs.pooler_output)
        return self._normalize(features).tolist()[0]

    def embed_image_url(self, url: str) -> list[float]:
        """Fetch an image from URL and embed it."""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return self.embed_image(image)

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        """Embed an image from raw bytes (e.g., file upload)."""
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.embed_image(image)

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        features = features.cpu().float()
        return features / features.norm(p=2, dim=-1, keepdim=True)


# Module-level singleton
clip_service = CLIPEmbeddingService()
