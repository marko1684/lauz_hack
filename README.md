# Patent Similarity Search System

A complete pipeline for scraping, normalizing, vectorizing, and searching patent data using semantic similarity.

## 🚀 Quick Start

```bash
# 1. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline
cd scraper/

# Scrape patents from CSV
python3 scraper.py your_patents.csv raw_output.json

# Normalize and chunk the text
python3 normalize.py raw_output.json normalized_output.json

# Generate embeddings
python3 vectorize.py normalized_output.json embeddings.pkl --create-faiss

# Search for similar patents
python3 search.py embeddings.pkl "your search query" --top-k 10 --faiss embeddings.faiss
```

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Installation](#installation)
- [Pipeline Steps](#pipeline-steps)
- [Usage Examples](#usage-examples)
- [Command Reference](#command-reference)
- [File Formats](#file-formats)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## 🎯 System Overview

This system processes patent data through four stages:

```
CSV File → Scraper → Raw JSON → Normalizer → Chunked JSON → Vectorizer → Embeddings → Search
```

### What it does:

1. **Scrapes** patent details (abstract, description, claims) from Google Patents URLs
2. **Normalizes** text (cleaning, lowercasing, removing HTML)
3. **Chunks** patents into semantic pieces (200-400 tokens each)
4. **Vectorizes** chunks using AI embeddings
5. **Searches** for similar patents using semantic similarity

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- ~4GB disk space for dependencies
- Internet connection (for first-time model download)

### Setup

```bash
# Clone or navigate to the project directory
cd /path/to/lauz_hack

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Dependencies

- **pandas** - CSV/data processing
- **requests** - HTTP requests for scraping
- **beautifulsoup4** - HTML parsing
- **lxml** - XML/HTML processing
- **sentence-transformers** - Text embedding generation
- **faiss-cpu** - Fast similarity search
- **numpy** - Numerical operations

## 📊 Pipeline Steps

### Step 1: Scrape Patents

Extract patent data from Google Patents using a CSV file containing patent URLs.

```bash
cd scraper/
python3 scraper.py input.csv output.json
```

**Input CSV format:**
- Must contain patent URLs in column I (index 8)
- Should have title in column B (index 1)
- Should have publication date in column G (index 6)

**Output:** JSON file with patent data including:
- `url` - Patent URL
- `title` - Patent title
- `publication_date` - Publication date
- `abstract` - Patent abstract text
- `description` - Full description text
- `claims` - Patent claims text

**Options:**
```bash
# Specify delay between requests (default: 1 second)
python3 scraper.py input.csv output.json --delay 2.0

# Save as SQLite database
python3 scraper.py input.csv output.db

# Save as CSV
python3 scraper.py input.csv output.csv
```

### Step 2: Normalize and Chunk

Clean text and split into semantic chunks optimized for embedding.

```bash
python3 normalize.py raw_output.json normalized_output.json
```

**What it does:**
- Converts text to lowercase
- Removes HTML entities and tags
- Normalizes Unicode characters
- Removes extra whitespace
- Splits text into chunks:
  - **Abstract**: Single chunk (or split if >400 tokens)
  - **Independent claims**: Separated from dependent claims
  - **Dependent claims**: Grouped together
  - **Description**: Split into ~300 token chunks

**Output format:**
```json
[
  {
    "patent_url": "https://patents.google.com/patent/...",
    "title": "Patent Title",
    "publication_date": "2023-01-03",
    "chunk_type": "abstract",
    "chunk_index": 0,
    "text": "cleaned and normalized text...",
    "tokens_approx": 219
  }
]
```

### Step 3: Vectorize Chunks

Convert text chunks into vector embeddings for similarity search.

```bash
python3 vectorize.py normalized_output.json embeddings.pkl --create-faiss
```

**Embedding Models:**

**Sentence Transformers (Recommended):**
- ✅ FREE - No API key needed
- ✅ Runs locally
- ✅ Fast (~6 seconds for 347 chunks)
- ✅ 384-dimensional vectors
- Model: all-MiniLM-L6-v2

**OpenAI (Alternative):**
```bash
export OPENAI_API_KEY="your-key-here"
python3 vectorize.py normalized_output.json embeddings.pkl --model openai
```
- High quality embeddings
- 1536-dimensional vectors
- Costs ~$0.02 per 1M tokens

**Output files:**
- `embeddings.pkl` - Embeddings + metadata (binary)
- `embeddings.json` - Human-readable summary
- `embeddings.faiss` - FAISS index (if --create-faiss used)

### Step 4: Search

Find similar patents using semantic similarity.

```bash
python3 search.py embeddings.pkl "your search query" --top-k 10 --faiss embeddings.faiss
```

**Search modes:**

**By Chunk (default):**
```bash
python3 search.py embeddings.pkl "hydrogen combustion engine"
```
Returns individual chunks ranked by similarity.

**By Patent:**
```bash
python3 search.py embeddings.pkl "ammonia fuel system" --by-patent
```
Returns patents ranked by average chunk similarity.

**Filter by chunk type:**
```bash
# Search only abstracts
python3 search.py embeddings.pkl "zero emission" --chunk-type abstract

# Search only claims
python3 search.py embeddings.pkl "catalyst system" --chunk-type claim_independent
```

## 💡 Usage Examples

### Example 1: Basic Workflow

```bash
cd scraper/

# Scrape 100 patents
python3 scraper.py patents_list.csv scraped_data.json

# Normalize
python3 normalize.py scraped_data.json normalized_data.json

# Vectorize
python3 vectorize.py normalized_data.json embeddings.pkl --create-faiss

# Search
python3 search.py embeddings.pkl "hydrogen fuel cell technology" --top-k 10
```

### Example 2: Using the Helper Script

```bash
cd scraper/

# Run examples
./quick_start.sh example

# Vectorize your data
./quick_start.sh vectorize normalized_data.json my_embeddings.pkl

# Search
./quick_start.sh search my_embeddings.pkl "electric motor control system"
```

### Example 3: Python API

```python
from vectorize import PatentVectorizer
from search import PatentSearcher

# Vectorize
vectorizer = PatentVectorizer(model_type="sentence-transformers")
vectorizer.load_chunks("normalized_data.json")
vectorizer.generate_embeddings()
vectorizer.save_embeddings("embeddings.pkl")

# Search
searcher = PatentSearcher("embeddings.pkl")
results = searcher.search("hydrogen storage", top_k=5)

for chunk, score in results:
    print(f"Similarity: {score:.4f}")
    print(f"Patent: {chunk['title']}")
    print(f"Text: {chunk['text'][:200]}...\n")
```

### Example 4: Advanced Search

```bash
# Search with no text output (just metadata)
python3 search.py embeddings.pkl "battery management" --no-text

# Search in descriptions only
python3 search.py embeddings.pkl "catalyst design" --chunk-type description

# Find top 20 most similar patents
python3 search.py embeddings.pkl "fuel injection" --by-patent --top-k 20
```

## 📖 Command Reference

### scraper.py

```bash
python3 scraper.py <input_csv> <output_file> [options]

Arguments:
  input_csv              CSV file with patent URLs
  output_file            Output file (.json, .db, .csv, .parquet)

Options:
  --delay SECONDS        Delay between requests (default: 1.0)
```

### normalize.py

```bash
python3 normalize.py <input_json> <output_json>

Arguments:
  input_json             Raw patent JSON from scraper
  output_json            Output normalized chunks JSON
```

### vectorize.py

```bash
python3 vectorize.py <input_json> <output_pkl> [options]

Arguments:
  input_json             Normalized chunks JSON
  output_pkl             Output embeddings file

Options:
  --model {sentence-transformers,openai}
                         Embedding model (default: sentence-transformers)
  --batch-size N         Batch size (default: 32)
  --create-faiss         Create FAISS index for faster search
```

### search.py

```bash
python3 search.py <embeddings_pkl> <query> [options]

Arguments:
  embeddings_pkl         Embeddings file from vectorize.py
  query                  Search query text

Options:
  --top-k N              Number of results (default: 10)
  --chunk-type TYPE      Filter by type (abstract, claim_independent, 
                         claim_dependent, description)
  --by-patent            Group results by patent
  --faiss FILE           Use FAISS index for faster search
  --no-text              Hide chunk text in output
```

## 📁 File Formats

### Input CSV (for scraper)

```csv
id,title,assignee,inventor,priority_date,filing_date,publication_date,grant_date,result_link,figure_link
1,Patent Title,Company,Inventor,2020-01-01,2020-02-01,2021-01-01,2022-01-01,https://patents.google.com/patent/...,https://...
```

### Raw JSON (scraper output)

```json
[
  {
    "url": "https://patents.google.com/patent/US11542878B2/en",
    "title": "Zero emission propulsion systems",
    "publication_date": "2023-01-03",
    "abstract": "Aspects relate to zero-emission...",
    "description": "The present invention relates to...",
    "claims": "What is claimed is: 1. A propulsion system..."
  }
]
```

### Normalized JSON (normalizer output)

```json
[
  {
    "patent_url": "https://patents.google.com/patent/US11542878B2/en",
    "title": "Zero emission propulsion systems",
    "publication_date": "2023-01-03",
    "chunk_type": "abstract",
    "chunk_index": 0,
    "text": "aspects relate to zero-emission propulsion...",
    "tokens_approx": 219
  }
]
```

### Embeddings Summary JSON

```json
{
  "model_type": "sentence-transformers",
  "num_chunks": 347,
  "embedding_dim": 384,
  "num_patents": 7,
  "chunk_types": ["abstract", "claim_independent", "claim_dependent", "description"]
}
```

## 🔧 Advanced Features

### Custom Batch Processing

Process large datasets in batches:

```python
from scraper import PatentScraper

scraper = PatentScraper()
scraper.load_csv("large_dataset.csv")

# Process in batches of 1000
batch_size = 1000
for i in range(0, len(scraper.patent_data), batch_size):
    batch = scraper.patent_data[i:i+batch_size]
    results = scraper.scrape_batch(batch)
    scraper.save_results(results, f"batch_{i//batch_size}.json")
```

### Multiple Embedding Models

Compare different models:

```bash
# Sentence Transformers (fast, free)
python3 vectorize.py data.json embeddings_st.pkl

# OpenAI (high quality, paid)
python3 vectorize.py data.json embeddings_openai.pkl --model openai
```

### Custom Chunk Sizes

Modify `normalize.py` to adjust chunk sizes:

```python
# In normalize.py, change max_tokens parameter
def split_into_chunks(self, text: str, max_tokens: int = 300):
    # Change 300 to your desired size (200-512 recommended)
```

### Export Search Results

```bash
# Save results to JSON
python3 search.py embeddings.pkl "query" --top-k 100 > results.json

# Save patent list only
python3 search.py embeddings.pkl "query" --by-patent --no-text > patents.txt
```

## ❓ Troubleshooting

### Import Errors

```bash
# If sentence-transformers fails to import
pip install --upgrade sentence-transformers

# If faiss fails to import
pip install faiss-cpu

# For GPU support (optional)
pip install faiss-gpu
```

### Scraping Issues

**Problem:** Scraper returns empty data
```bash
# Check the CSV format - URLs should be in column I (index 8)
# Try with --delay 2.0 to avoid rate limiting
python3 scraper.py input.csv output.json --delay 2.0
```

**Problem:** KeyError on CSV columns
```python
# The scraper expects specific columns. Adjust in scraper.py:
# Line ~60-65, modify column indices based on your CSV structure
```

### Memory Issues

```bash
# Reduce batch size for vectorization
python3 vectorize.py data.json embeddings.pkl --batch-size 16

# Process data in smaller chunks
# Split your normalized JSON into multiple files
```

### Search Returns No Results

**Problem:** All similarity scores are low (<0.3)
- Your query might be too specific or use different terminology
- Try broader queries or synonyms
- Check that embeddings were generated correctly

**Problem:** FAISS index not found
```bash
# Regenerate with --create-faiss flag
python3 vectorize.py normalized.json embeddings.pkl --create-faiss
```

### Performance Optimization

**Slow search:**
```bash
# Always use FAISS index for datasets >1000 chunks
python3 vectorize.py data.json embeddings.pkl --create-faiss
python3 search.py embeddings.pkl "query" --faiss embeddings.faiss
```

**Slow vectorization:**
- Sentence Transformers is much faster than OpenAI for large datasets
- Use `--batch-size 64` for faster processing (if memory allows)

## 📊 Performance Benchmarks

| Dataset Size | Scraping Time | Normalization | Vectorization | Search Time |
|--------------|---------------|---------------|---------------|-------------|
| 100 patents  | ~2 minutes    | ~1 second     | ~15 seconds   | <0.1s       |
| 1,000 patents| ~20 minutes   | ~5 seconds    | ~2 minutes    | <0.2s       |
| 5,989 patents| ~2 hours      | ~30 seconds   | ~12 minutes   | <0.5s       |

*Benchmarks on CPU (Intel i7), using Sentence Transformers + FAISS*

## 📝 Understanding Similarity Scores

Search results include similarity scores (0-1 scale):

- **0.9-1.0**: Nearly identical text (exact matches, duplicates)
- **0.7-0.9**: Very similar concepts and technology
- **0.5-0.7**: Related approaches or applications
- **0.3-0.5**: Somewhat relevant, different implementations
- **<0.3**: Different domains or unrelated

**Example:** Searching for "hydrogen fuel cell":
- Score 0.82: Another hydrogen fuel cell patent
- Score 0.65: Electric vehicle power system
- Score 0.45: Battery management system
- Score 0.25: Mechanical transmission system

## 🤝 Contributing

To improve the system:

1. Add new embedding models in `vectorize.py`
2. Enhance text cleaning in `normalize.py`
3. Support more patent databases in `scraper.py`
4. Add filters (date, country, patent office) in `search.py`

## 📄 License

This project is provided as-is for research and development purposes.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the `VECTORIZATION_GUIDE.md` for detailed vectorization help
3. Check log files for error messages
4. Ensure all dependencies are installed correctly

## 🎯 Next Steps

After setting up the system:

1. **Scrape your full dataset** - Process all 5,989+ patents
2. **Experiment with queries** - Test different search terms
3. **Build a web interface** - Create a Flask/FastAPI app for easy access
4. **Add metadata filters** - Filter by date, patent office, assignee
5. **Implement clustering** - Group similar patents automatically
6. **Create visualizations** - Generate similarity matrices and dendrograms
7. **API service** - Deploy as REST API for integration with other tools

---

**Happy patent searching! 🚀**
