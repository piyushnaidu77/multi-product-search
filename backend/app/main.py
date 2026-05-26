from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import search, products, health
from app.core.config import settings

app = FastAPI(
    title="Multimodal Product Search API",
    description="Search products by image or text using CLIP embeddings + Pinecone",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(products.router, prefix="/api/products", tags=["products"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
