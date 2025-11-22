# Patent Vectorization and Similarity Search Guide

## Overview

This guide explains how to vectorize your patent data and perform similarity searches.

## Pipeline

```
CSV → scraper.py → JSON (raw) → normalize.py → JSON (chunks) → vectorize.py → Embeddings → search.py → Results
```

## Installation

Install the required packages:

```bash
pip install sentence-transformers faiss-cpu numpy
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Step 1: Vectorize Your Data

Convert normalized patent chunks into vector embeddings:

```bash
# Using Sentence Transformers (FREE, local, no API key needed)
python3 vectorize.py normalized_test.json embeddings.pkl

# With FAISS index for faster search
python3 vectorize.py normalized_test.json embeddings.pkl --create-faiss

# Using OpenAI embeddings (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-api-key-here"
python3 vectorize.py normalized_test.json embeddings.pkl --model openai
```

### Output Files

- `embeddings.pkl` - Contains embeddings + chunk metadata
- `embeddings.json` - Human-readable summary
- `embeddings.faiss` - FAISS index for fast search (if --create-faiss used)

## Step 2: Search for Similar Patents

### Basic Search

```bash
# Search for similar chunks
python3 search.py embeddings.pkl "hydrogen combustion engine with low emissions"
```

### Advanced Search Options

```bash
# Top 20 results
python3 search.py embeddings.pkl "ammonia fuel injection" --top-k 20

# Search only in abstracts
python3 search.py embeddings.pkl "zero emission propulsion" --chunk-type abstract

# Search only in claims
python3 search.py embeddings.pkl "catalyst system" --chunk-type claim_independent

# Group results by patent (not by chunk)
python3 search.py embeddings.pkl "hydrogen storage system" --by-patent

# Use FAISS for faster search
python3 search.py embeddings.pkl "fuel cell system" --faiss embeddings.faiss

# Hide text in output (just show metadata)
python3 search.py embeddings.pkl "combustion chamber design" --no-text
```

## Search from Python Code

You can also use the vectorization and search functionality in your own Python code:

```python
from vectorize import PatentVectorizer
from search import PatentSearcher

# Vectorize data
vectorizer = PatentVectorizer(model_type="sentence-transformers")
vectorizer.load_chunks("normalized_test.json")
vectorizer.generate_embeddings()
vectorizer.save_embeddings("embeddings.pkl")

# Search
searcher = PatentSearcher("embeddings.pkl")
results = searcher.search("hydrogen engine", top_k=10)

for chunk, score in results:
    print(f"Similarity: {score:.4f}")
    print(f"Patent: {chunk['title']}")
    print(f"Text: {chunk['text'][:200]}...")
    print()
```

## Embedding Models Comparison

### Sentence Transformers (Recommended for most users)

**Model: all-MiniLM-L6-v2**
- ✅ FREE - No API key needed
- ✅ Runs locally - No internet required after download
- ✅ Fast - ~14ms per chunk on CPU
- ✅ Small - 384-dimensional vectors
- ✅ Good quality - Trained on 1B+ sentence pairs
- ⚠️ Download size: ~80MB

**Alternative: all-mpnet-base-v2**
- Better quality but slower
- 768-dimensional vectors
- Change in vectorize.py: `SentenceTransformer('all-mpnet-base-v2')`

### OpenAI Embeddings

**Model: text-embedding-3-small**
- ✅ High quality embeddings
- ✅ 1536-dimensional vectors
- ❌ Requires API key
- ❌ Costs money (~$0.02 per 1M tokens)
- ❌ Requires internet connection
- ⚠️ Rate limits apply

For 347 chunks (~85K tokens): ~$0.002

## Full Workflow Example

```bash
# 1. Scrape patents (if not done yet)
python3 scraper.py gp-search-20251122-081501.csv patents.json

# 2. Normalize and chunk
python3 normalize.py patents.json normalized_patents.json

# 3. Generate embeddings
python3 vectorize.py normalized_patents.json embeddings.pkl --create-faiss

# 4. Search
python3 search.py embeddings.pkl "hydrogen fuel cell" --top-k 10 --faiss embeddings.faiss
```

## Understanding Results

Results are sorted by **cosine similarity** (0-1 scale):
- **0.9-1.0**: Nearly identical text
- **0.7-0.9**: Very similar concepts/technology
- **0.5-0.7**: Related but different approaches
- **<0.5**: Somewhat relevant or different domains

## Performance Tips

1. **Use FAISS** for datasets >1000 chunks
2. **Use sentence-transformers** if you don't need OpenAI-level quality
3. **Filter by chunk_type** to search only abstracts or claims
4. **Use --by-patent** to find similar patents rather than chunks
5. **Batch process** if vectorizing large datasets

## Troubleshooting

### Import Error: sentence_transformers

```bash
pip install sentence-transformers
```

### Import Error: faiss

```bash
pip install faiss-cpu
# Or for GPU support:
pip install faiss-gpu
```

### OpenAI API Error

```bash
export OPENAI_API_KEY="your-key-here"
```

### Out of Memory

- Reduce batch size: `--batch-size 16`
- Use a smaller model
- Process in smaller chunks

## Next Steps

1. **Scale up**: Vectorize your full dataset (5,989 patents)
2. **Build a web interface**: Create a search UI
3. **Add filters**: Search by date, patent type, etc.
4. **Semantic clustering**: Group similar patents together
5. **API service**: Create REST API for patent search
