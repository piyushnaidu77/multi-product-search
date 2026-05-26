"""
Dataset Ingestion Script
------------------------
Indexes the local Amazon Berkeley Objects (ABO) dataset into Pinecone.

Expects the extracted abo-images-small layout:
    <data-dir>/small/<id[:2]>/<id>.jpg
    <data-dir>/listings/metadata/listings_*.json.gz

Usage:
    python -m scripts.ingest_dataset
    python -m scripts.ingest_dataset --limit 5000 --category SHOES
    python -m scripts.ingest_dataset --data-dir C:/path/to/data/images
"""

import argparse
import gzip
import json
import logging
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.clip_service import clip_service
from app.services.pinecone_service import pinecone_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "images"


def _pick_english(values: list) -> str:
    """Return first English-tagged value, falling back to the first entry."""
    if not values:
        return ""
    for v in values:
        if str(v.get("language_tag", "")).startswith("en"):
            return v.get("value", "")
    return values[0].get("value", "")


def load_image_map(data_dir: Path) -> dict:
    """Load images.csv.gz into a {image_id: relative_path} dict."""
    csv_path = data_dir / "metadata" / "images.csv.gz"
    logger.info(f"Loading image map from {csv_path.name} ...")
    image_map = {}
    with gzip.open(csv_path, "rt", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                image_map[parts[0]] = parts[3]  # image_id -> path
    logger.info(f"Loaded {len(image_map):,} image entries")
    return image_map


def iter_listings(data_dir: Path):
    """Yield one listing dict per line from all listings_*.json.gz files."""
    listings_dir = data_dir / "listings" / "metadata"
    gz_files = sorted(listings_dir.glob("listings_*.json.gz"))
    if not gz_files:
        raise FileNotFoundError(f"No listings_*.json.gz found in {listings_dir}")
    for gz_path in gz_files:
        logger.info(f"Reading {gz_path.name}")
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def ingest(limit: int = 5000, category_filter: str = "all", data_dir: Path = DEFAULT_DATA_DIR):
    """
    Reads ABO listings and images from disk, generates CLIP embeddings,
    and bulk-upserts into Pinecone.
    """
    logger.info(f"Data directory: {data_dir}")
    image_map = load_image_map(data_dir)

    vectors = []
    skipped = 0
    processed = 0

    for item in tqdm(iter_listings(data_dir), total=limit, desc="Embedding products"):
        if processed >= limit:
            break

        try:
            item_id = item.get("item_id", f"product_{processed}")
            image_id = item.get("main_image_id")
            if not image_id or image_id not in image_map:
                skipped += 1
                continue

            img_path = data_dir / "small" / image_map[image_id]
            if not img_path.exists():
                skipped += 1
                continue

            product_type_list = item.get("product_type", [])
            category_str = product_type_list[0].get("value", "general") if product_type_list else "general"

            if category_filter != "all" and category_filter.lower() not in category_str.lower():
                continue

            name = _pick_english(item.get("item_name", [])) or "Unknown Product"

            image = Image.open(img_path).convert("RGB")
            embedding = clip_service.embed_image(image)

            metadata = {
                "name": name[:100],
                "category": category_str[:50],
                "image_url": f"https://m.media-amazon.com/images/I/{image_id}.jpg",
                "description": name[:200],
            }

            vectors.append({"id": item_id, "values": embedding, "metadata": metadata})
            processed += 1

            if len(vectors) >= 100:
                pinecone_service.upsert_batch(vectors)
                vectors = []

        except Exception as e:
            logger.warning(f"Skipped item at index {processed}: {e}")
            skipped += 1

    if vectors:
        pinecone_service.upsert_batch(vectors)

    stats = pinecone_service.get_stats()
    logger.info(f"Done. Indexed: {processed}, Skipped: {skipped}")
    logger.info(f"Pinecone index stats: {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000, help="Max products to index")
    parser.add_argument("--category", type=str, default="all", help="Filter by product_type (e.g. SHOES)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to ABO images directory (must contain small/ and listings/)",
    )
    args = parser.parse_args()
    ingest(limit=args.limit, category_filter=args.category, data_dir=args.data_dir)
