"""
Search Router
-------------
POST /api/search/text   — search by text query
POST /api/search/image  — search by uploaded image
POST /api/search/url    — search by image URL
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from typing import Optional
from app.services.clip_service import clip_service
from app.services.pinecone_service import pinecone_service
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ──────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 12
    category: Optional[str] = None


class ImageUrlRequest(BaseModel):
    image_url: HttpUrl
    top_k: int = 12
    category: Optional[str] = None


class SearchResult(BaseModel):
    id: str
    score: float
    name: str
    category: str
    price: Optional[float]
    image_url: str
    description: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_type: str
    total: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_filter(category: Optional[str]) -> Optional[dict]:
    if category:
        return {"category": {"$eq": category}}
    return None


def _format_response(results: list[dict], query_type: str) -> SearchResponse:
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query_type=query_type,
        total=len(results),
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/text", response_model=SearchResponse)
async def search_by_text(body: TextSearchRequest):
    """
    Search the product catalog using a natural language query.
    Example: "red running shoes for women"
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    embedding = clip_service.embed_text(body.query)
    results = pinecone_service.query(
        embedding=embedding,
        top_k=body.top_k,
        filter=_build_filter(body.category),
    )
    return _format_response(results, "text")


@router.post("/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(default=12, le=50),
    category: Optional[str] = Query(default=None),
):
    """
    Search by uploading an image.
    Accepts: JPEG, PNG, WEBP. Max size: 5MB.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image must be under 5MB.")

    embedding = clip_service.embed_image_bytes(contents)
    results = pinecone_service.query(
        embedding=embedding,
        top_k=top_k,
        filter=_build_filter(category),
    )
    return _format_response(results, "image")


@router.post("/url", response_model=SearchResponse)
async def search_by_image_url(body: ImageUrlRequest):
    """
    Search by providing a public image URL.
    Useful for searching from existing product pages.
    """
    try:
        embedding = clip_service.embed_image_url(str(body.image_url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")

    results = pinecone_service.query(
        embedding=embedding,
        top_k=body.top_k,
        filter=_build_filter(body.category),
    )
    return _format_response(results, "image_url")
