#!/usr/bin/env python3
"""
Patent Similarity Search Script
Search for similar patent chunks using vector embeddings.
"""

import json
import logging
import argparse
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatentSearcher:
    """Search for similar patents using vector embeddings."""
    
    def __init__(self, embeddings_file: str, model_path: str = None):
        """
        Initialize the searcher.
        
        Args:
            embeddings_file: Path to the embeddings pickle file
            model_path: Optional path to fine-tuned model directory (overrides default model)
        """
        self.embeddings_file = embeddings_file
        self.model_path = model_path
        self.embeddings = None
        self.chunks = None
        self.model_type = None
        self.model = None
        self.faiss_index = None
        
        self._load_embeddings()
        self._initialize_model()
    
    def _load_embeddings(self):
        """Load embeddings and metadata from file."""
        logger.info(f"Loading embeddings from {self.embeddings_file}")
        
        with open(self.embeddings_file, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.chunks = data['chunks']
        self.model_type = data['model_type']
        
        logger.info(f"Loaded {len(self.chunks)} chunks with {data['embedding_dim']}-dimensional embeddings")
    
    def _initialize_model(self):
        """Initialize the embedding model for query encoding."""
        if self.model_type == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                
                # Use fine-tuned model if provided, otherwise use default
                if self.model_path:
                    logger.info(f"Loading fine-tuned model from {self.model_path}...")
                    self.model = SentenceTransformer(self.model_path)
                else:
                    logger.info("Loading default Sentence Transformer model...")
                    # Temporarily using all-MiniLM-L6-v2 to match existing embeddings
                    # TODO: Re-vectorize with all-mpnet-base-v2 for better quality
                    self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
                
        elif self.model_type == "openai":
            try:
                import openai
                import os
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                openai.api_key = api_key
                self.model = openai
            except ImportError:
                logger.error("openai not installed")
                raise
    
    def load_faiss_index(self, faiss_file: str):
        """Load a pre-built FAISS index for faster search."""
        try:
            import faiss
            logger.info(f"Loading FAISS index from {faiss_file}")
            self.faiss_index = faiss.read_index(faiss_file)
            logger.info(f"FAISS index loaded with {self.faiss_index.ntotal} vectors")
        except ImportError:
            logger.warning("faiss not installed, will use numpy for search")
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")
    
    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a query text into an embedding vector."""
        if self.model_type == "sentence-transformers":
            embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
            
        elif self.model_type == "openai":
            response = self.model.embeddings.create(
                model="text-embedding-3-small",
                input=[query_text]
            )
            embedding = np.array(response.data[0].embedding)
        
        return embedding
    
    def search(self, query_text: str, top_k: int = 10, 
               chunk_type: str = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar patent chunks.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            chunk_type: Optional filter by chunk type (abstract, claim_independent, etc.)
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity
        """
        logger.info(f"Searching for: '{query_text[:100]}...'")
        
        # Encode query
        query_embedding = self.encode_query(query_text)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # Filter by chunk type if specified
        if chunk_type:
            indices = [i for i, c in enumerate(self.chunks) if c['chunk_type'] == chunk_type]
            search_embeddings = self.embeddings[indices]
            search_chunks = [self.chunks[i] for i in indices]
        else:
            search_embeddings = self.embeddings
            search_chunks = self.chunks
            indices = list(range(len(self.chunks)))
        
        # Normalize embeddings
        search_embeddings_norm = search_embeddings / np.linalg.norm(search_embeddings, axis=1, keepdims=True)
        
        # Use FAISS if available, otherwise numpy
        if self.faiss_index and not chunk_type:
            # FAISS search (fast)
            scores, result_indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k
            )
            results = [
                (self.chunks[idx], float(score)) 
                for idx, score in zip(result_indices[0], scores[0])
            ]
        else:
            # Numpy search (slower but flexible)
            similarities = np.dot(search_embeddings_norm, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [
                (search_chunks[idx], float(similarities[idx])) 
                for idx in top_indices
            ]
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def display_results(self, results: List[Tuple[Dict[str, Any], float]], 
                       show_text: bool = True, max_text_len: int = 300):
        """Display search results in a readable format."""
        print("\n" + "="*80)
        print(f"TOP {len(results)} RESULTS")
        print("="*80 + "\n")
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"[{i}] Similarity: {score:.4f}")
            print(f"    Patent: {chunk['title']}")
            print(f"    URL: {chunk['patent_url']}")
            print(f"    Publication Date: {chunk['publication_date']}")
            print(f"    Chunk Type: {chunk['chunk_type']}")
            print(f"    Tokens: {chunk['tokens_approx']}")
            
            if show_text:
                text = chunk['text']
                if len(text) > max_text_len:
                    text = text[:max_text_len] + "..."
                print(f"    Text: {text}")
            
            print()
    
    def find_similar_patents(self, query_text: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Find most similar patents (grouped by patent_url).
        
        Returns:
            List of (patent_url, title, avg_similarity_score) tuples
        """
        # Get all chunks
        all_results = self.search(query_text, top_k=len(self.chunks))
        
        # Group by patent
        patent_scores = {}
        for chunk, score in all_results:
            url = chunk['patent_url']
            title = chunk['title']
            if url not in patent_scores:
                patent_scores[url] = {'title': title, 'scores': []}
            patent_scores[url]['scores'].append(score)
        
        # Calculate average score per patent
        patent_results = [
            (url, data['title'], np.mean(data['scores']))
            for url, data in patent_scores.items()
        ]
        
        # Sort by average score
        patent_results.sort(key=lambda x: x[2], reverse=True)
        
        return patent_results[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Search for similar patent chunks')
    parser.add_argument('embeddings_file', help='Input embeddings file (.pkl)')
    parser.add_argument('query', help='Search query text')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to return (default: 10)')
    parser.add_argument('--chunk-type', choices=['abstract', 'claim_independent', 
                                                  'claim_dependent', 'description'],
                       help='Filter by chunk type')
    parser.add_argument('--no-text', action='store_true',
                       help='Hide chunk text in results')
    parser.add_argument('--faiss', help='Path to FAISS index file (.faiss)')
    parser.add_argument('--by-patent', action='store_true',
                       help='Group results by patent instead of by chunk')
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = PatentSearcher(args.embeddings_file)
    
    # Load FAISS index if provided
    if args.faiss:
        searcher.load_faiss_index(args.faiss)
    
    # Search
    if args.by_patent:
        results = searcher.find_similar_patents(args.query, top_k=args.top_k)
        print("\n" + "="*80)
        print(f"TOP {len(results)} SIMILAR PATENTS")
        print("="*80 + "\n")
        for i, (url, title, score) in enumerate(results, 1):
            print(f"[{i}] Avg Similarity: {score:.4f}")
            print(f"    Title: {title}")
            print(f"    URL: {url}")
            print()
    else:
        results = searcher.search(args.query, top_k=args.top_k, chunk_type=args.chunk_type)
        searcher.display_results(results, show_text=not args.no_text)


if __name__ == '__main__':
    main()
