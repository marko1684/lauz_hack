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
    global searcher, embeddings_path, faiss_path
    
    # Default paths - can be configured via environment variables
    embeddings_path = os.getenv(
        "EMBEDDINGS_PATH",
        str(Path(__file__).parent.parent / "scraper" / "test_embeddings.pkl")
    )
    faiss_path = os.getenv(
        "FAISS_PATH",
        str(Path(__file__).parent.parent / "scraper" / "test_embeddings.faiss")
    )
    
    try:
        logger.info(f"Loading embeddings from: {embeddings_path}")
        searcher = PatentSearcher(embeddings_path)
        
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
        # Perform search for each chunk type if specified
        if submission.chunk_types:
            all_results = []
            for chunk_type in submission.chunk_types:
                results = searcher.search(
                    submission.text,
                    top_k=submission.top_k,
                    chunk_type=chunk_type
                )
                all_results.extend(results)
            
            # Sort by similarity and take top_k
            all_results.sort(key=lambda x: x[1], reverse=True)
            results = all_results[:submission.top_k]
        else:
            # Search all chunks
            results = searcher.search(
                submission.text,
                top_k=submission.top_k * 2  # Get more results for filtering
            )
        
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
                ]
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
                ]
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
