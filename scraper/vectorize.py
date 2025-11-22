#!/usr/bin/env python3
"""
Patent Vectorization Script
Converts normalized patent chunks into vector embeddings for similarity search.
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatentVectorizer:
    """Vectorizes patent chunks using embedding models."""
    
    def __init__(self, model_type: str = "sentence-transformers"):
        """
        Initialize the vectorizer.
        
        Args:
            model_type: Type of embedding model to use:
                - "sentence-transformers" (free, local, good for general text)
                - "openai" (API-based, requires API key, high quality)
        """
        self.model_type = model_type
        self.model = None
        self.embeddings = []
        self.chunks = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on model_type."""
        if self.model_type == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                # Using all-mpnet-base-v2: Better quality for technical/patent text
                # Alternative: all-MiniLM-L6-v2 (384 dim, faster but less accurate)
                # For patents, better model = better discrimination
                logger.info("Loading Sentence Transformer model (all-mpnet-base-v2)...")
                self.model = SentenceTransformer('all-mpnet-base-v2')
                logger.info("Model loaded successfully")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
                
        elif self.model_type == "openai":
            try:
                import openai
                import os
                # Requires OPENAI_API_KEY environment variable
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                openai.api_key = api_key
                logger.info("OpenAI API key configured")
                self.model = openai
            except ImportError:
                logger.error("openai not installed. Install with: pip install openai")
                raise
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def load_chunks(self, json_file: str) -> List[Dict[str, Any]]:
        """Load normalized patent chunks from JSON file."""
        logger.info(f"Loading chunks from {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks")
        self.chunks = chunks
        return chunks
    
    def generate_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for all chunks.
        
        Args:
            batch_size: Number of chunks to process at once (for efficiency)
            
        Returns:
            Array of embeddings, shape (num_chunks, embedding_dim)
        """
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")
        
        texts = [chunk['text'] for chunk in self.chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        if self.model_type == "sentence-transformers":
            # Sentence Transformers processes in batches automatically
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
        elif self.model_type == "openai":
            # OpenAI has rate limits, process in batches with delays
            import time
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                try:
                    response = self.model.embeddings.create(
                        model="text-embedding-3-small",  # 1536 dimensions
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    # Rate limiting: wait a bit between batches
                    if i + batch_size < len(texts):
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                    raise
            
            embeddings = np.array(embeddings)
        
        self.embeddings = embeddings
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, output_file: str):
        """
        Save embeddings and metadata to file.
        
        Saves as a pickle file containing:
        - embeddings: numpy array
        - chunks: list of chunk metadata
        - model_type: string
        """
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings generated. Call generate_embeddings() first.")
        
        output_path = Path(output_file)
        logger.info(f"Saving embeddings to {output_file}")
        
        data = {
            'embeddings': self.embeddings,
            'chunks': self.chunks,
            'model_type': self.model_type,
            'embedding_dim': self.embeddings.shape[1],
            'num_chunks': len(self.chunks)
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {len(self.embeddings)} embeddings")
        logger.info(f"Embedding dimension: {self.embeddings.shape[1]}")
        
        # Also save a human-readable summary
        summary_file = output_path.with_suffix('.json')
        summary = {
            'model_type': self.model_type,
            'num_chunks': len(self.chunks),
            'embedding_dim': int(self.embeddings.shape[1]),
            'num_patents': len(set(c['patent_url'] for c in self.chunks)),
            'chunk_types': list(set(c['chunk_type'] for c in self.chunks))
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")
    
    def create_faiss_index(self, output_file: str = None):
        """
        Create a FAISS index for fast similarity search.
        
        FAISS (Facebook AI Similarity Search) is optimized for nearest neighbor search.
        """
        try:
            import faiss
        except ImportError:
            logger.error("faiss not installed. Install with: pip install faiss-cpu")
            return
        
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings generated. Call generate_embeddings() first.")
        
        logger.info("Creating FAISS index...")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Create index
        dimension = embeddings_normalized.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity after normalization)
        index.add(embeddings_normalized.astype('float32'))
        
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        
        if output_file:
            faiss.write_index(index, output_file)
            logger.info(f"FAISS index saved to {output_file}")
        
        return index


def main():
    parser = argparse.ArgumentParser(description='Vectorize patent chunks for similarity search')
    parser.add_argument('input_json', help='Input JSON file with normalized chunks')
    parser.add_argument('output_file', help='Output file for embeddings (.pkl)')
    parser.add_argument('--model', choices=['sentence-transformers', 'openai'], 
                       default='sentence-transformers',
                       help='Embedding model to use (default: sentence-transformers)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation (default: 32)')
    parser.add_argument('--create-faiss', action='store_true',
                       help='Also create a FAISS index for fast search')
    
    args = parser.parse_args()
    
    # Create vectorizer
    vectorizer = PatentVectorizer(model_type=args.model)
    
    # Load chunks
    vectorizer.load_chunks(args.input_json)
    
    # Generate embeddings
    vectorizer.generate_embeddings(batch_size=args.batch_size)
    
    # Save embeddings
    vectorizer.save_embeddings(args.output_file)
    
    # Optionally create FAISS index
    if args.create_faiss:
        faiss_file = Path(args.output_file).with_suffix('.faiss')
        vectorizer.create_faiss_index(str(faiss_file))
    
    logger.info("Vectorization complete!")


if __name__ == '__main__':
    main()
