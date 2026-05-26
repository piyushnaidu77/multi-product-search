"""
Products Router
---------------
POST /api/products        — add a single product to the index
DELETE /api/products/{id} — remove a product
GET  /api/products/stats  — index statistics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional
from app.services.clip_service import clip_service
from app.services.pinecone_service import pinecone_service

router = APIRouter()


class ProductCreate(BaseModel):
    id: str
    name: str
    category: str
    price: Optional[float] = None
    image_url: HttpUrl
    description: str = ""


@router.post("/", status_code=201)
async def add_product(product: ProductCreate):
    """
    Index a new product. Fetches its image, generates a CLIP
    embedding, and upserts it into Pinecone.
    """
    try:
        embedding = clip_service.embed_image_url(str(product.image_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not embed image: {e}")

    metadata = {
        "name": product.name,
        "category": product.category,
        "price": product.price,
        "image_url": str(product.image_url),
        "description": product.description,
    }
    pinecone_service.upsert_product(product.id, embedding, metadata)
    return {"message": "Product indexed successfully", "id": product.id}


@router.delete("/{product_id}")
async def delete_product(product_id: str):
    pinecone_service.delete_product(product_id)
    return {"message": "Product removed", "id": product_id}


@router.get("/stats")
async def index_stats():
    return pinecone_service.get_stats()
