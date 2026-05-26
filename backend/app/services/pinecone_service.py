"""
Pinecone Vector DB Service
--------------------------
Handles all interactions with Pinecone: upsert, query, delete.
Index dimension must match CLIP output (512 for ViT-B/32).
Uses cosine similarity metric for normalized embeddings.
"""

from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
import logging
from typing import Optional

logger = logging.getLogger(__name__)

VECTOR_DIMENSION = 512  # CLIP ViT-B/32 output dim


class PineconeService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.index(settings.PINECONE_INDEX_NAME)
        self._initialized = True
        logger.info(f"Pinecone ready — index: {settings.PINECONE_INDEX_NAME}")

    def _ensure_index(self):
        """Create the index if it doesn't exist yet."""
        existing = [i.name for i in self.pc.indexes.list()]
        if settings.PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating index '{settings.PINECONE_INDEX_NAME}'...")
            self.pc.indexes.create(
                name=settings.PINECONE_INDEX_NAME,
                dimension=VECTOR_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # free tier
            )

    def upsert_product(self, product_id: str, embedding: list[float], metadata: dict):
        """
        Insert or update a product vector.
        metadata keys: name, category, price, image_url, description
        """
        self.index.upsert(vectors=[{
            "id": product_id,
            "values": embedding,
            "metadata": metadata,
        }])

    def upsert_batch(self, vectors: list[dict], batch_size: int = 100):
        """
        Bulk upsert for indexing large catalogs.
        vectors: list of {"id": str, "values": list[float], "metadata": dict}
        """
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    def query(
        self,
        embedding: list[float],
        top_k: int = None,
        filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Find the top-k most similar products.
        Returns list of {id, score, metadata} dicts.
        Optional filter: {"category": {"$eq": "shoes"}}
        """
        top_k = top_k or settings.TOP_K_RESULTS
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter,
        )
        return [
            {
                "id": match.id,
                "score": round(match.score, 4),
                "name": match.metadata.get("name", ""),
                "category": match.metadata.get("category", ""),
                "price": match.metadata.get("price"),
                "image_url": match.metadata.get("image_url", ""),
                "description": match.metadata.get("description", ""),
            }
            for match in results.matches
            if match.score >= settings.MIN_SIMILARITY_SCORE
        ]

    def delete_product(self, product_id: str):
        self.index.delete(ids=[product_id])

    def get_stats(self) -> dict:
        return self.index.describe_index_stats()


pinecone_service = PineconeService()
