# Multi-Product Search

A multimodal product search platform that finds products using natural language text, image uploads, or image URLs. Powered by CLIP embeddings and Pinecone vector search.

## How It Works

CLIP (Contrastive Language-Image Pre-training) encodes both text and images into the same 512-dimensional vector space, enabling cross-modal semantic similarity search. Queries and product images are embedded and compared using cosine similarity in Pinecone.

**Supported search modes:**
- **Text** — describe what you're looking for (e.g. "red running shoes for women")
- **Image upload** — drag & drop or click to upload a product image
- **Image URL** — paste a link to an existing product image

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.11, Uvicorn |
| Embeddings | CLIP (`openai/clip-vit-base-patch32`) via HuggingFace Transformers |
| Vector DB | Pinecone (serverless, cosine similarity) |
| Optional DB | PostgreSQL 16 + SQLAlchemy + Alembic |
| Containers | Docker + Docker Compose |

## Project Structure

```
multi-product-search/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── core/config.py       # Environment & settings
│   │   ├── api/                 # Route handlers (search, products, health)
│   │   └── services/            # CLIP and Pinecone singletons
│   ├── scripts/
│   │   └── ingest_dataset.py    # Bulk product ingestion from ABO dataset
│   └── tests/                   # pytest unit & integration tests
├── frontend/
│   └── src/
│       ├── app/page.tsx         # Main search page
│       ├── components/          # SearchBar, ResultsGrid, ProductCard
│       ├── hooks/useSearch.ts   # Search state management
│       └── utils/api.ts         # Axios API client
├── data/images/                 # Local product images
└── docker-compose.yml
```

## Prerequisites

- [Docker & Docker Compose](https://docs.docker.com/get-docker/) — recommended
- Or: Python 3.11+ and Node.js 18+ for local development
- A [Pinecone](https://www.pinecone.io/) account and API key (free tier works)

## Setup

### 1. Configure environment variables

Create `backend/.env`:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=product-search
PINECONE_ENVIRONMENT=gcp-starter

CLIP_MODEL_NAME=openai/clip-vit-base-patch32

APP_ENV=development
ALLOWED_ORIGINS=["http://localhost:3000"]

TOP_K_RESULTS=12
MIN_SIMILARITY_SCORE=0.20

# Optional — for extended product metadata
DATABASE_URL=postgresql://user:password@localhost:5432/productsearch

# Optional — for S3 image storage
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_BUCKET_NAME=
AWS_REGION=us-east-1
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### 2. Run with Docker

```bash
docker-compose up
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### 3. Run locally (without Docker)

**Backend:**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

## Ingest Products

Load the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) dataset or your own product catalog:

```bash
cd backend
python -m scripts.ingest_dataset --limit 5000 --category SHOES --data-dir ./data
```

Options:
- `--limit` — number of products to ingest (default: all)
- `--category` — filter by product category
- `--data-dir` — path to local dataset directory

To add a single product via API:

```bash
curl -X POST http://localhost:8000/api/products/ \
  -H "Content-Type: application/json" \
  -d '{"id": "product-1", "name": "Running Shoe", "image_url": "https://..."}'
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/search/text` | Search by text query |
| `POST` | `/api/search/image` | Search by image upload (JPEG/PNG/WEBP, max 5MB) |
| `POST` | `/api/search/url` | Search by image URL |
| `POST` | `/api/products/` | Add a product to the index |
| `DELETE` | `/api/products/{id}` | Remove a product |
| `GET` | `/api/products/stats` | Index statistics |

Full interactive docs available at `http://localhost:8000/docs`.

## Running Tests

```bash
cd backend
pytest
pytest tests/test_search_api.py -v   # specific file
```

Tests use mocks for CLIP and Pinecone — no model downloads or API calls required.

## Frontend Scripts

```bash
npm run dev     # Development server
npm run build   # Production build
npm run start   # Production server
npm run lint    # ESLint
```
