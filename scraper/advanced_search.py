#!/usr/bin/env python3
"""
Advanced Patent Search with Better Filtering and Scoring
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from search import PatentSearcher
import logging

logger = logging.getLogger(__name__)


class AdvancedPatentSearcher(PatentSearcher):
    """Enhanced patent searcher with better scoring and filtering."""
    
    def __init__(self, embeddings_file: str, similarity_threshold: float = 0.0):
        """
        Initialize advanced searcher.
        
        Args:
            embeddings_file: Path to embeddings file
            similarity_threshold: Minimum similarity score to return (0.0-1.0)
        """
        super().__init__(embeddings_file)
        self.similarity_threshold = similarity_threshold
    
    def search_with_threshold(
        self, 
        query_text: str, 
        top_k: int = 10,
        min_similarity: float = 0.5,
        chunk_type: str = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search with minimum similarity threshold.
        
        Args:
            query_text: Search query
            top_k: Max results to return
            min_similarity: Minimum similarity score (e.g., 0.5 = 50%)
            chunk_type: Optional chunk type filter
            
        Returns:
            Filtered results above threshold
        """
        # Get initial results
        results = self.search(query_text, top_k=top_k * 2, chunk_type=chunk_type)
        
        # Filter by threshold
        filtered_results = [
            (chunk, score) for chunk, score in results 
            if score >= min_similarity
        ]
        
        return filtered_results[:top_k]
    
    def search_with_reranking(
        self,
        query_text: str,
        top_k: int = 10,
        chunk_type: str = None,
        boost_exact_match: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search with re-ranking based on keyword overlap.
        
        This helps surface results that share important technical terms.
        """
        results = self.search(query_text, top_k=top_k * 3, chunk_type=chunk_type)
        
        if not boost_exact_match:
            return results[:top_k]
        
        # Extract key terms from query (simple tokenization)
        query_terms = set(query_text.lower().split())
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_terms = query_terms - stopwords
        
        # Re-score based on term overlap
        rescored_results = []
        for chunk, score in results:
            chunk_terms = set(chunk['text'].lower().split())
            
            # Calculate term overlap bonus
            overlap = len(query_terms & chunk_terms)
            overlap_ratio = overlap / len(query_terms) if query_terms else 0
            
            # Boost score by up to 20% based on overlap
            boosted_score = score * (1 + 0.2 * overlap_ratio)
            boosted_score = min(boosted_score, 1.0)  # Cap at 1.0
            
            rescored_results.append((chunk, boosted_score))
        
        # Re-sort by boosted scores
        rescored_results.sort(key=lambda x: x[1], reverse=True)
        
        return rescored_results[:top_k]
    
    def search_by_section(
        self,
        query_text: str,
        top_k_per_section: int = 5,
        min_similarity: float = 0.5
    ) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """
        Search each patent section separately.
        
        Returns:
            Dict with results grouped by chunk_type
        """
        sections = ['abstract', 'claim_independent', 'claim_dependent', 'description']
        results_by_section = {}
        
        for section in sections:
            results = self.search_with_threshold(
                query_text,
                top_k=top_k_per_section,
                min_similarity=min_similarity,
                chunk_type=section
            )
            if results:
                results_by_section[section] = results
        
        return results_by_section
    
    def calculate_diversity_penalty(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        penalty_factor: float = 0.1
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Penalize results from the same patent to promote diversity.
        
        Args:
            results: Initial search results
            penalty_factor: How much to reduce score for duplicates (0.0-1.0)
            
        Returns:
            Re-scored results with diversity
        """
        seen_patents = {}
        diverse_results = []
        
        for chunk, score in results:
            patent_url = chunk['patent_url']
            
            if patent_url in seen_patents:
                # Apply penalty for duplicate patent
                seen_patents[patent_url] += 1
                penalty = penalty_factor * seen_patents[patent_url]
                adjusted_score = score * (1 - penalty)
            else:
                seen_patents[patent_url] = 0
                adjusted_score = score
            
            diverse_results.append((chunk, adjusted_score))
        
        # Re-sort by adjusted scores
        diverse_results.sort(key=lambda x: x[1], reverse=True)
        
        return diverse_results


def main():
    """Example usage of advanced search features."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced patent search')
    parser.add_argument('embeddings_file', help='Embeddings file')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--min-similarity', type=float, default=0.5,
                       help='Minimum similarity threshold (0.0-1.0)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results')
    parser.add_argument('--rerank', action='store_true',
                       help='Use re-ranking with keyword boost')
    parser.add_argument('--by-section', action='store_true',
                       help='Search each section separately')
    
    args = parser.parse_args()
    
    searcher = AdvancedPatentSearcher(args.embeddings_file)
    
    if args.by_section:
        results = searcher.search_by_section(
            args.query,
            top_k_per_section=args.top_k,
            min_similarity=args.min_similarity
        )
        
        for section, section_results in results.items():
            print(f"\n{'='*80}")
            print(f"SECTION: {section.upper()}")
            print(f"{'='*80}\n")
            
            for i, (chunk, score) in enumerate(section_results, 1):
                print(f"[{i}] Similarity: {score:.4f} ({score*100:.1f}%)")
                print(f"    Patent: {chunk['title']}")
                print(f"    URL: {chunk['patent_url']}")
                print()
    
    elif args.rerank:
        results = searcher.search_with_reranking(
            args.query,
            top_k=args.top_k
        )
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"[{i}] Similarity: {score:.4f} ({score*100:.1f}%)")
            print(f"    Patent: {chunk['title']}")
            print(f"    Type: {chunk['chunk_type']}")
            print(f"    Text: {chunk['text'][:150]}...")
            print()
    
    else:
        results = searcher.search_with_threshold(
            args.query,
            top_k=args.top_k,
            min_similarity=args.min_similarity
        )
        
        if not results:
            print(f"No results found above {args.min_similarity*100:.0f}% similarity threshold")
        else:
            for i, (chunk, score) in enumerate(results, 1):
                print(f"[{i}] Similarity: {score:.4f} ({score*100:.1f}%)")
                print(f"    Patent: {chunk['title']}")
                print(f"    Type: {chunk['chunk_type']}")
                print(f"    Text: {chunk['text'][:150]}...")
                print()


if __name__ == '__main__':
    main()
