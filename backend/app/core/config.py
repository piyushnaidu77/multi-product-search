from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "product-search"
    PINECONE_ENVIRONMENT: str = "gcp-starter"  # free tier

    # CLIP model
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # AWS S3 (optional - for product images)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_BUCKET_NAME: str = ""
    AWS_REGION: str = "us-east-1"

    # PostgreSQL
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/productsearch"

    # Search defaults
    TOP_K_RESULTS: int = 12
    MIN_SIMILARITY_SCORE: float = 0.20

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
