# Patent Similarity Search Backend API

FastAPI-based REST API for patent similarity search.

## Quick Start

### 1. Install Dependencies

```bash
# From the lauz_hack directory
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 2. Start the Server

```bash
# Development mode (auto-reload)
cd backend
python api.py

# Or using uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the API

- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### POST /search

Search for similar patents.

**Request:**
```json
{
  "text": "A hydrogen combustion engine with low emissions and high efficiency",
  "title": "Hydrogen Engine",
  "chunk_types": ["abstract", "claim_independent"],
  "top_k": 10,
  "by_patent": false
}
```

**Response:**
```json
{
  "query": "A hydrogen combustion engine...",
  "timestamp": "2025-11-22T19:00:00",
  "processing_time_ms": 45.2,
  "results_count": 10,
  "chunks": [
    {
      "patent_url": "https://patents.google.com/patent/US123456",
      "title": "Hydrogen Fuel System",
      "publication_date": "2023-01-03",
      "chunk_type": "abstract",
      "chunk_index": 0,
      "text": "A system for hydrogen combustion...",
      "tokens_approx": 245,
      "similarity_score": 0.8234
    }
  ]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "embeddings_loaded": true,
  "num_chunks": 347,
  "embedding_dim": 384
}
```

### GET /stats

Get database statistics.

**Response:**
```json
{
  "total_chunks": 347,
  "unique_patents": 7,
  "chunk_types": {
    "abstract": 7,
    "claim_independent": 140,
    "claim_dependent": 50,
    "description": 150
  },
  "embedding_dimension": 384,
  "model_type": "sentence-transformers",
  "faiss_enabled": true
}
```

## Configuration

### Environment Variables

```bash
# Path to embeddings file
export EMBEDDINGS_PATH=/path/to/embeddings.pkl

# Path to FAISS index (optional, but recommended for speed)
export FAISS_PATH=/path/to/embeddings.faiss
```

### CORS Configuration

Edit `api.py` to configure allowed origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourfrontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Frontend Integration

### JavaScript/React Example

```javascript
// Search for similar patents
async function searchPatents(patentText) {
  const response = await fetch('http://localhost:8000/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: patentText,
      title: 'My Patent',
      top_k: 10,
      by_patent: false
    })
  });
  
  const data = await response.json();
  return data;
}

// Usage
const results = await searchPatents("hydrogen combustion engine");
console.log(results.chunks);
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A hydrogen combustion engine with zero emissions",
    "top_k": 5,
    "by_patent": true
  }'
```

## Deployment

### Development

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Production with Gunicorn

```bash
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

```bash
# Build
docker build -t patent-search-api .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/../scraper/embeddings.pkl:/app/data/embeddings.pkl \
  -v $(pwd)/../scraper/embeddings.faiss:/app/data/embeddings.faiss \
  patent-search-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./scraper/embeddings.pkl:/app/data/embeddings.pkl
      - ./scraper/embeddings.faiss:/app/data/embeddings.faiss
    environment:
      - EMBEDDINGS_PATH=/app/data/embeddings.pkl
      - FAISS_PATH=/app/data/embeddings.faiss
    restart: unless-stopped
```

## Performance Optimization

### 1. Use FAISS Index

Always generate and use FAISS index for fast search:

```bash
python vectorize.py normalized.json embeddings.pkl --create-faiss
```

### 2. Increase Workers

For production, use multiple workers:

```bash
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 3. Add Caching (Optional)

For frequently searched queries, add Redis caching:

```python
import redis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# In startup event
redis_client = redis.Redis(host='localhost', port=6379, db=0)
FastAPICache.init(RedisBackend(redis_client), prefix="patent-search:")
```

### 4. Load Balancing

Use Nginx or similar for load balancing multiple API instances.

## Testing

### Manual Testing

Visit http://localhost:8000/docs to test endpoints interactively.

### Automated Testing

```python
# test_api.py
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_search():
    response = client.post("/search", json={
        "text": "hydrogen engine",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "chunks" in data
    assert len(data["chunks"]) <= 5

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

Run tests:
```bash
pytest test_api.py
```

## Monitoring

### Logging

Logs are written to stdout/stderr. In production, configure logging:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/patent-api.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics

Add Prometheus metrics:

```bash
pip install prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Access metrics at http://localhost:8000/metrics

## Security

### Production Checklist

- [ ] Configure CORS properly (don't use `allow_origins=["*"]`)
- [ ] Add API key authentication
- [ ] Rate limiting (use slowapi)
- [ ] Input validation (already using Pydantic)
- [ ] HTTPS only
- [ ] Hide /reload endpoint or add authentication

### Adding API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Use in endpoints
@app.post("/search", dependencies=[Security(verify_api_key)])
async def search_similar_patents(submission: PatentSubmission):
    ...
```

## Troubleshooting

### Server won't start

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use a different port
uvicorn api:app --port 8001
```

### Embeddings not loading

```bash
# Check environment variables
echo $EMBEDDINGS_PATH

# Verify file exists
ls -lh /path/to/embeddings.pkl

# Check logs for error messages
```

### Slow search performance

- Ensure FAISS index is loaded
- Check if running on CPU vs GPU
- Increase number of workers
- Consider caching results

## License

This API is part of the Patent Similarity Search System.
