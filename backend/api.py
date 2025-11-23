#!/usr/bin/env python3
"""
FastAPI Backend for Patent Similarity Search
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add scraper directory to path
sys.path.append(str(Path(__file__).parent.parent / 'scraper'))

from search import PatentSearcher
from vectorize import PatentVectorizer
from scraper import PatentScraper
from normalize import PatentNormalizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Patent Similarity Search API",
    description="API for finding similar patents using semantic search",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
searcher: Optional[PatentSearcher] = None
normalizer: Optional[PatentNormalizer] = None
embeddings_path: Optional[str] = None
faiss_path: Optional[str] = None


# Pydantic models for request/response
class PatentSubmission(BaseModel):
    """Model for submitted patent claim/text"""
    text: str = Field(..., min_length=10, max_length=100000, description="Patent text to search (up to 100KB)")
    title: Optional[str] = Field(None, max_length=1000, description="Optional patent title")
    chunk_types: Optional[List[str]] = Field(
        None, 
        description="Filter by chunk types (abstract, claim_independent, claim_dependent, description)"
    )
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    by_patent: bool = Field(False, description="Group results by patent instead of chunks")
    use_chunking: bool = Field(False, description="Apply semantic chunking to input text")


class ChunkRequest(BaseModel):
    """Model for text chunking request"""
    text: str = Field(..., min_length=10, max_length=100000, description="Text to chunk")
    max_tokens: int = Field(350, ge=50, le=1000, description="Maximum tokens per chunk")
    sim_threshold: float = Field(0.10, ge=0.0, le=1.0, description="Similarity threshold for semantic grouping")


class ChunkResponse(BaseModel):
    """Model for chunked text response"""
    chunks: List[str]
    num_chunks: int
    processing_time_ms: float


class SimilarChunk(BaseModel):
    """Model for a similar patent chunk result"""
    patent_url: str
    title: str
    publication_date: str
    chunk_type: str
    chunk_index: Any
    text: str
    tokens_approx: int
    similarity_score: float


class SimilarPatent(BaseModel):
    """Model for a similar patent result (grouped)"""
    patent_url: str
    title: str
    avg_similarity: float
    publication_date: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    timestamp: str
    processing_time_ms: float
    results_count: int
    chunks: Optional[List[SimilarChunk]] = None
    patents: Optional[List[SimilarPatent]] = None
    query_chunks: Optional[List[str]] = None  # The chunked query text if chunking was used


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    embeddings_loaded: bool
    num_chunks: Optional[int] = None
    embedding_dim: Optional[int] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


class SentenceMatch(BaseModel):
    """Model for a sentence-level match"""
    your_sentence: str
    matched_sentence: str
    similarity_score: float
    section: str  # abstract, description, or claims


class DetailedComparisonRequest(BaseModel):
    """Request model for detailed comparison"""
    your_text: str = Field(..., description="Your patent text")
    matched_patent_url: str = Field(..., description="URL of the matched patent to compare against")


class DetailedComparisonResponse(BaseModel):
    """Response model for detailed comparison"""
    your_sentences: List[Dict[str, Any]]  # Each sentence with its best match info
    matched_sentences: List[Dict[str, Any]]  # Each matched patent sentence with match info
    processing_time_ms: float


class FullPatent(BaseModel):
    """Model for full scraped patent"""
    url: str
    title: str
    publication_date: Optional[str] = None
    abstract: str
    description: str
    claims: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the searcher on startup"""
    global searcher, normalizer, embeddings_path, faiss_path
    
    # Default paths - can be configured via environment variables
    embeddings_path = os.getenv(
        "EMBEDDINGS_PATH",
        str(Path(__file__).parent.parent / "scraper" / "full_embeddings.pkl")
    )
    faiss_path = os.getenv(
        "FAISS_PATH",
        str(Path(__file__).parent.parent / "scraper" / "full_embeddings.faiss")
    )
    model_path = os.getenv(
        "MODEL_PATH",
        None  # Will use default model if not specified
    )
    
    try:
        logger.info(f"Loading embeddings from: {embeddings_path}")
        if model_path:
            logger.info(f"Using fine-tuned model from: {model_path}")
        searcher = PatentSearcher(embeddings_path, model_path=model_path)
        
        # Initialize normalizer for semantic chunking
        logger.info("Initializing patent normalizer for semantic chunking...")
        normalizer = PatentNormalizer("")  # Empty path, we'll just use chunking methods
        logger.info("Normalizer initialized successfully")
        
        # Load FAISS index if available
        if Path(faiss_path).exists():
            logger.info(f"Loading FAISS index from: {faiss_path}")
            searcher.load_faiss_index(faiss_path)
        else:
            logger.warning("FAISS index not found, will use numpy search (slower)")
        
        logger.info("Searcher initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize searcher: {e}")
        # Don't crash the app, just log the error


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Patent Similarity Search API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if searcher is None:
        return HealthResponse(
            status="error",
            embeddings_loaded=False
        )
    
    return HealthResponse(
        status="ok",
        embeddings_loaded=True,
        num_chunks=len(searcher.chunks) if searcher.chunks else None,
        embedding_dim=searcher.embeddings.shape[1] if searcher.embeddings is not None else None
    )


@app.post("/chunk", response_model=ChunkResponse, tags=["Preprocessing"])
async def chunk_text(request: ChunkRequest):
    """
    Chunk text using semantic similarity-based splitting.
    Groups sentences by semantic coherence while respecting token limits.
    
    - **text**: Text to chunk
    - **max_tokens**: Maximum tokens per chunk (50-1000)
    - **sim_threshold**: Similarity threshold for grouping sentences (0.0-1.0)
    """
    if normalizer is None:
        raise HTTPException(
            status_code=503,
            detail="Normalizer not initialized. Check server logs."
        )
    
    start_time = datetime.now()
    
    try:
        # Clean and chunk the text
        cleaned_text = normalizer.clean_text(request.text)
        chunks = normalizer.semantic_split(
            cleaned_text,
            max_tokens=request.max_tokens,
            sim_threshold=request.sim_threshold
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        return ChunkResponse(
            chunks=chunks,
            num_chunks=len(chunks),
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_similar_patents(
    submission: PatentSubmission,
    min_similarity: float = 0.0  # Add threshold parameter
):
    """
    Search for similar patents based on submitted text.
    
    - **text**: Patent claim or description text to search
    - **title**: Optional title for the submission
    - **chunk_types**: Optional list to filter by chunk types
    - **top_k**: Number of results to return (1-100)
    - **by_patent**: Group results by patent instead of individual chunks
    - **min_similarity**: Minimum similarity threshold (0.0-1.0, default: 0.0)
    """
    if searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Search service not initialized. Check server logs."
        )
    
    start_time = datetime.now()
    
    try:
        # Apply fast chunking if requested (uses simple sentence-based splitting)
        # Clean text with boilerplate removal to reduce false matches
        search_texts = [submission.text]
        query_chunks = None
        if submission.use_chunking and normalizer is not None:
            cleaned_text = normalizer.clean_text(submission.text, remove_boilerplate=True)
            chunks = normalizer.simple_split(cleaned_text, max_tokens=350)
            if len(chunks) > 1:
                search_texts = chunks
                query_chunks = chunks  # Store for response
                logger.info(f"Chunked input text into {len(chunks)} chunks")
        elif normalizer is not None:
            # Even without chunking, clean the text to remove boilerplate
            cleaned_text = normalizer.clean_text(submission.text, remove_boilerplate=True)
            if cleaned_text:
                search_texts = [cleaned_text]
        
        # Aggregate results from all chunks with score averaging
        chunk_scores = {}  # {chunk_key: [list of scores]}
        chunk_data = {}    # {chunk_key: chunk dict}
        
        for text_chunk in search_texts:
            # Perform search for each chunk type if specified
            if submission.chunk_types:
                for chunk_type in submission.chunk_types:
                    results = searcher.search(
                        text_chunk,
                        top_k=submission.top_k * 3,  # Get more results for better averaging
                        chunk_type=chunk_type
                    )
                    for chunk, score in results:
                        chunk_key = (chunk['patent_url'], chunk['chunk_type'], chunk['chunk_index'])
                        if chunk_key not in chunk_scores:
                            chunk_scores[chunk_key] = []
                            chunk_data[chunk_key] = chunk
                        chunk_scores[chunk_key].append(score)
            else:
                # Search all chunks
                results = searcher.search(
                    text_chunk,
                    top_k=submission.top_k * 3  # Get more results for better averaging
                )
                for chunk, score in results:
                    chunk_key = (chunk['patent_url'], chunk['chunk_type'], chunk['chunk_index'])
                    if chunk_key not in chunk_scores:
                        chunk_scores[chunk_key] = []
                        chunk_data[chunk_key] = chunk
                    chunk_scores[chunk_key].append(score)
        
        # Calculate average scores (or max if only seen once)
        all_results = []
        for chunk_key, scores in chunk_scores.items():
            # Average score across all input chunks that matched this result
            avg_score = sum(scores) / len(scores) if len(search_texts) > 1 else scores[0]
            all_results.append((chunk_data[chunk_key], avg_score))
        
        # Sort by average similarity and take top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        results = all_results
        
        # Apply similarity threshold filter
        if min_similarity > 0.0:
            results = [(chunk, score) for chunk, score in results if score >= min_similarity]
        
        results = results[:submission.top_k]  # Trim to requested size
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        if submission.by_patent:
            # Group by patent
            patent_results = searcher.find_similar_patents(
                submission.text,
                top_k=submission.top_k
            )
            
            return SearchResponse(
                query=submission.text[:200] + "..." if len(submission.text) > 200 else submission.text,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=round(processing_time, 2),
                results_count=len(patent_results),
                patents=[
                    SimilarPatent(
                        patent_url=url,
                        title=title,
                        avg_similarity=round(score, 4)
                    )
                    for url, title, score in patent_results
                ],
                query_chunks=query_chunks
            )
        else:
            # Return individual chunks
            return SearchResponse(
                query=submission.text[:200] + "..." if len(submission.text) > 200 else submission.text,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=round(processing_time, 2),
                results_count=len(results),
                chunks=[
                    SimilarChunk(
                        patent_url=chunk['patent_url'],
                        title=chunk['title'],
                        publication_date=chunk['publication_date'],
                        chunk_type=chunk['chunk_type'],
                        chunk_index=chunk['chunk_index'],
                        text=chunk['text'],
                        tokens_approx=chunk['tokens_approx'],
                        similarity_score=round(score, 4)
                    )
                    for chunk, score in results
                ],
                query_chunks=query_chunks
            )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/stats", tags=["Info"])
async def get_statistics():
    """Get statistics about the loaded patent database"""
    if searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Search service not initialized"
        )
    
    # Calculate statistics
    unique_patents = set(chunk['patent_url'] for chunk in searcher.chunks)
    chunk_types = {}
    for chunk in searcher.chunks:
        chunk_type = chunk['chunk_type']
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    return {
        "total_chunks": len(searcher.chunks),
        "unique_patents": len(unique_patents),
        "chunk_types": chunk_types,
        "embedding_dimension": searcher.embeddings.shape[1],
        "model_type": searcher.model_type,
        "faiss_enabled": searcher.faiss_index is not None
    }


@app.post("/reload", tags=["Admin"])
async def reload_embeddings(background_tasks: BackgroundTasks):
    """
    Reload embeddings from disk (admin endpoint).
    Use this after updating the embeddings file.
    """
    global searcher
    
    if searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Search service not initialized"
        )
    
    def reload_task():
        global searcher
        try:
            logger.info("Reloading embeddings...")
            searcher = PatentSearcher(embeddings_path)
            if Path(faiss_path).exists():
                searcher.load_faiss_index(faiss_path)
            logger.info("Embeddings reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload embeddings: {e}")
    
    background_tasks.add_task(reload_task)
    
    return {
        "message": "Reloading embeddings in background",
        "status": "pending"
    }


@app.get("/patent/{patent_url:path}", response_model=FullPatent, tags=["Patents"])
async def scrape_full_patent(patent_url: str):
    """
    Scrape and return the full patent content from Google Patents.
    This provides complete text for detailed comparison.
    
    Args:
        patent_url: Full URL to the patent on Google Patents
    
    Returns:
        FullPatent object with all sections
    """
    try:
        logger.info(f"Scraping full patent: {patent_url}")
        
        # Initialize scraper (create dummy csv path since we're not using it)
        scraper = PatentScraper(csv_path="dummy.csv")
        
        # Create patent_data dict as expected by scrape_patent_details
        patent_data = {
            'url': patent_url,
            'title': 'Patent',  # Will be extracted from the page
            'publication_date': None  # Will be extracted from the page
        }
        
        # Scrape the patent
        result = scraper.scrape_patent_details(patent_data)
        
        if not result or result.get('error'):
            raise HTTPException(
                status_code=404,
                detail=f"Could not scrape patent: {result.get('error', 'Unknown error')}"
            )
        
        # Check if we got any content
        if not result.get('abstract') and not result.get('description') and not result.get('claims'):
            raise HTTPException(
                status_code=404,
                detail="No content found in patent"
            )
        
        return FullPatent(
            url=result['url'],
            title=result.get('title', 'Unknown Title'),
            publication_date=result.get('publication_date'),
            abstract=result.get('abstract', ''),
            description=result.get('description', ''),
            claims=result.get('claims', '')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scraping patent: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scrape patent: {str(e)}"
        )


@app.post("/compare", response_model=DetailedComparisonResponse, tags=["Comparison"])
async def detailed_comparison(request: DetailedComparisonRequest):
    """
    Perform detailed sentence-by-sentence comparison between your patent and a matched patent.
    This uses a cross-encoder model for more accurate pairwise comparison.
    """
    import time
    start_time = time.time()
    
    try:
        from sentence_transformers import CrossEncoder
        
        # Initialize cross-encoder (only once, could be cached)
        logger.info("Loading cross-encoder model for detailed comparison...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Scrape the matched patent
        scraper = PatentScraper(csv_path="dummy.csv")
        patent_data = {'url': request.matched_patent_url, 'title': 'Patent', 'publication_date': None}
        result = scraper.scrape_patent_details(patent_data)
        
        if not result or result.get('error'):
            raise HTTPException(status_code=404, detail="Could not scrape matched patent")
        
        # Clean junk from text
        import re
        def remove_junk(text):
            cleaned_text = text

            cleaned_text = re.sub(
                r'\b(?:fig|figure|img|image|FIG|FIGURE|IMG|IMAGE|Figs|FIGS)\.?\s*\d+\b',
                '', cleaned_text,
                flags=re.IGNORECASE
            )

            cleaned_text = re.sub(r'\[\s*\d+\s*\]', '', cleaned_text)
            cleaned_text = re.sub(r'\(\s*(?:\d+|[ivx]+|[a-zA-Z])\s*\)', '', cleaned_text, flags=re.IGNORECASE)
            
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
            cleaned_text = cleaned_text.strip()

            return cleaned_text
        
        def split_sentences(text):
            sentences = re.split(r'[.!?]+\s+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Clean input text before processing
        cleaned_your_text = remove_junk(request.your_text)
        your_sentences = split_sentences(cleaned_your_text)
        
        # Clean and combine matched patent sections
        matched_sections = {
            'abstract': split_sentences(remove_junk(result.get('abstract', ''))),
            'description': split_sentences(remove_junk(result.get('description', '')))[:50],  # Limit description
            'claims': split_sentences(remove_junk(result.get('claims', '')))
        }
        
        # Flatten all matched sentences
        all_matched_sentences = []
        sentence_to_section = {}
        for section, sentences in matched_sections.items():
            for sent in sentences:
                all_matched_sentences.append(sent)
                sentence_to_section[sent] = section
        
        logger.info(f"Stage 1: Pre-filtering with bi-encoder - {len(your_sentences)} your sentences vs {len(all_matched_sentences)} matched sentences")
        
        # Stage 1: Use bi-encoder to quickly pre-filter candidates (much faster)
        from sentence_transformers import SentenceTransformer, util
        import torch
        
        # Use the same model as the searcher for consistency
        bi_encoder = searcher.model if searcher else SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Encode all sentences
        your_embeddings = bi_encoder.encode(your_sentences, convert_to_tensor=True, show_progress_bar=False)
        matched_embeddings = bi_encoder.encode(all_matched_sentences, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute cosine similarities (very fast)
        cosine_scores = util.cos_sim(your_embeddings, matched_embeddings)
        
        # For each of your sentences, get top 10 candidate matches
        top_k = 10
        all_pairs = []
        pair_to_indices = {}
        
        for i, your_sent in enumerate(your_sentences):
            # Get top-k most similar matched sentences
            top_results = torch.topk(cosine_scores[i], k=min(top_k, len(all_matched_sentences)))
            
            for score, j in zip(top_results[0], top_results[1]):
                matched_sent = all_matched_sentences[j]
                pair_key = len(all_pairs)
                all_pairs.append([your_sent, matched_sent])
                pair_to_indices[pair_key] = (i, int(j))
        
        logger.info(f"Stage 2: Cross-encoder on {len(all_pairs)} pre-filtered pairs (reduced from {len(your_sentences) * len(all_matched_sentences)})")
        
        # Stage 2: Use cross-encoder only on pre-filtered pairs
        all_scores = cross_encoder.predict(all_pairs, batch_size=32, show_progress_bar=False)
        
        # Normalize scores to 0-1 range using sigmoid function
        # Cross-encoder returns logits, not probabilities
        import numpy as np
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        all_scores = sigmoid(np.array(all_scores))
        
        # Build score matrix from sparse results
        score_matrix = np.zeros((len(your_sentences), len(all_matched_sentences)))
        
        for pair_idx, (i, j) in pair_to_indices.items():
            score_matrix[i, j] = all_scores[pair_idx]
        
        # Build results from score matrix
        your_results = []
        for i, your_sent in enumerate(your_sentences):
            # Find best match for this sentence
            best_j = score_matrix[i].argmax()
            best_score = float(score_matrix[i, best_j])
            best_match = all_matched_sentences[best_j] if best_score > 0 else None
            best_section = sentence_to_section[all_matched_sentences[best_j]] if best_score > 0 else None
            
            your_results.append({
                'text': your_sent,
                'best_match': best_match,
                'similarity': best_score,
                'section': best_section
            })
        
        # For matched patent sentences, find best match from your text
        matched_results = []
        for j, matched_sent in enumerate(all_matched_sentences):
            best_i = score_matrix[:, j].argmax()
            best_score = float(score_matrix[best_i, j])
            best_your_sent = your_sentences[best_i] if best_score > 0.3 else None
            
            matched_results.append({
                'text': matched_sent,
                'section': sentence_to_section[matched_sent],
                'best_match': best_your_sent,
                'similarity': best_score
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetailedComparisonResponse(
            your_sentences=your_results,
            matched_sentences=matched_results,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python api.py
    # Or: uvicorn api:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
