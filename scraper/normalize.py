import json
import re
import html
import unicodedata
from typing import List, Dict
import logging
from sentence_transformers import SentenceTransformer, util

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PatentNormalizer:
    def __init__(self, input_json_path: str):
        """
        Initialize the patent normalizer.
        
        Args:
            input_json_path: Path to the scraped JSON file
        """
        self.input_path = input_json_path
        self.patents = []
        self.normalized_chunks = []
        # Initialize sentence transformer model once for all semantic splits
        logger.info("Loading sentence transformer model for semantic chunking...")
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        logger.info("Model loaded successfully")
        
    def load_patents(self):
        """Load patents from JSON file."""
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                self.patents = json.load(f)
            logger.info(f"Loaded {len(self.patents)} patents from {self.input_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading patents: {e}")
            return False
    
    def clean_text(self, text: str, remove_boilerplate: bool = False) -> str:
        """
        Clean text: lowercase, remove weird Unicode, strip HTML tags, normalize whitespace.
        
        Args:
            text: Raw text to clean
            remove_boilerplate: If True, remove common patent boilerplate text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize Unicode characters (remove accents, etc.)
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Lowercase
        text = text.lower()
        
        # Remove common patent boilerplate if requested
        if remove_boilerplate:
            # Remove figure references
            text = re.sub(r'\bfig\.?\s*\d+[a-z]?\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\bfigure\s+\d+[a-z]?\b', '', text, flags=re.IGNORECASE)
            
            # Remove reference numbers (e.g., "10", "20", "100" as standalone refs)
            text = re.sub(r'\b\d{1,3}\b(?=\s)', '', text)
            
            # Remove common patent phrases
            boilerplate_patterns = [
                r'\bherein\b',
                r'\bthereof\b',
                r'\btherefrom\b',
                r'\bthereby\b',
                r'\bwherein\b',
                r'\bhereby\b',
                r'\baforesaid\b',
                r'\bsaid\s+\w+\b',  # "said device", "said method"
                r'\babove-mentioned\b',
                r'\baforementioned\b',
                r'\baccordingly\b',
                r'\bpreferably\b',
                r'\bembodiment\b',
                r'\baspect\b',
                r'\bvariant\b',
                r'\bmodification\b',
                r'\bfurthermore\b',
                r'\bmoreover\b',
                r'\bin accordance with\b',
                r'\bwith respect to\b',
                r'\bat least one\b',
                r'\bone or more\b',
                r'\bplurality of\b',
            ]
            
            for pattern in boilerplate_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace (replace multiple spaces/newlines with single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def simple_split(self, text: str, max_tokens: int = 350) -> List[str]:
        """
        Fast sentence-based chunking without semantic analysis.
        Groups sentences by token count only - good for real-time use.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk (approximate)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [sentences[0]]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = len(sent) // 4
            
            if current_tokens + sent_tokens > max_tokens and current_chunk:
                # Save current chunk and start new
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent)
                current_tokens += sent_tokens
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def extract_claims(self, claims_text: str) -> Dict[str, List[str]]:
        """
        Parse and categorize claims into independent and dependent claims.
        
        Args:
            claims_text: Raw claims text
            
        Returns:
            Dictionary with 'independent' and 'dependent' claim lists
        """
        if not claims_text:
            return {'independent': [], 'dependent': []}
        
        # Split by claim numbers (e.g., "1.", "2.", "Claim 1", etc.)
        claim_pattern = r'(?:^|\n)\s*(?:claim\s+)?(\d+)\.?\s+'
        claims = re.split(claim_pattern, claims_text, flags=re.IGNORECASE)
        
        independent_claims = []
        dependent_claims = []
        
        # Process claims (claims list alternates: [text, num, claim_text, num, claim_text, ...])
        for i in range(1, len(claims), 2):
            if i + 1 < len(claims):
                claim_num = claims[i]
                claim_text = claims[i + 1].strip()
                
                if not claim_text:
                    continue
                
                # Check if it's a dependent claim (references another claim)
                if re.search(r'claim\s+\d+', claim_text, re.IGNORECASE) or \
                   re.search(r'claims?\s+\d+', claim_text, re.IGNORECASE) or \
                   'wherein' in claim_text.lower()[:50]:  # 'wherein' often indicates dependent claim
                    dependent_claims.append(f"claim {claim_num}: {claim_text}")
                else:
                    independent_claims.append(f"claim {claim_num}: {claim_text}")
        
        # If no claims were parsed, treat the whole text as one claim
        if not independent_claims and not dependent_claims:
            independent_claims.append(claims_text)
        
        return {
            'independent': independent_claims,
            'dependent': dependent_claims
        }
    
    def semantic_split(self, text: str, max_tokens: int = 350, sim_threshold: float = 0.10) -> List[str]:
        """
        Split text into semantically coherent chunks using sentence embeddings.
        Groups sentences by semantic similarity while respecting max_tokens limit.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk (approximate)
            sim_threshold: Minimum cosine similarity to keep sentences together (0.10 default)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # 1. Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [sentences[0]]
        
        # 2. Generate embeddings for all sentences
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_tokens = len(sentences[0]) // 4
        
        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_tokens = len(sent) // 4
            
            # Calculate similarity between current and previous sentence
            sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
            
            # If similarity high → same logical unit
            if sim >= sim_threshold:
                if current_tokens + sent_tokens > max_tokens:
                    # Would exceed max_tokens, save current chunk and start new
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_tokens = sent_tokens
                else:
                    # Add to current chunk
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
            else:
                # Low similarity → new semantic segment
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_tokens = sent_tokens
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def chunk_patent(self, patent: Dict) -> List[Dict]:
        """
        Create semantic chunks from a single patent.
        
        Args:
            patent: Patent dictionary with abstract, description, claims
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        patent_id = patent.get('url', 'unknown')
        title = patent.get('title', '')
        pub_date = patent.get('publication_date', '')
        
        # 1. Abstract chunk (single chunk - most important for overview)
        if patent.get('abstract'):
            abstract_clean = self.clean_text(patent['abstract'])
            if abstract_clean:
                # Split abstract if it's too long using semantic splitting
                abstract_tokens = len(abstract_clean) // 4
                if abstract_tokens > 400:
                    abstract_chunks = self.semantic_split(abstract_clean, max_tokens=350)
                    for idx, chunk_text in enumerate(abstract_chunks):
                        chunks.append({
                            'patent_url': patent_id,
                            'title': title,
                            'publication_date': pub_date,
                            'chunk_type': 'abstract',
                            'chunk_index': idx,
                            'text': chunk_text,
                            'tokens_approx': len(chunk_text) // 4
                        })
                else:
                    chunks.append({
                        'patent_url': patent_id,
                        'title': title,
                        'publication_date': pub_date,
                        'chunk_type': 'abstract',
                        'chunk_index': 0,
                        'text': abstract_clean,
                        'tokens_approx': len(abstract_clean) // 4
                    })
        
        # 2. Claims chunks (separated by type and importance)
        if patent.get('claims'):
            claims_clean = self.clean_text(patent['claims'])
            parsed_claims = self.extract_claims(claims_clean)
            
            # Independent claims (most important)
            # Split large claims into smaller chunks using semantic splitting
            for idx, claim in enumerate(parsed_claims['independent']):
                claim_clean = self.clean_text(claim)
                if claim_clean:
                    # If claim is too large, split it semantically
                    claim_tokens = len(claim_clean) // 4
                    if claim_tokens > 350:
                        # Split by semantic similarity
                        claim_sentences = self.semantic_split(claim_clean, max_tokens=300)
                        for sub_idx, claim_chunk in enumerate(claim_sentences):
                            chunks.append({
                                'patent_url': patent_id,
                                'title': title,
                                'publication_date': pub_date,
                                'chunk_type': 'claim_independent',
                                'chunk_index': f"{idx}_{sub_idx}",
                                'text': claim_chunk,
                                'tokens_approx': len(claim_chunk) // 4
                            })
                    else:
                        chunks.append({
                            'patent_url': patent_id,
                            'title': title,
                            'publication_date': pub_date,
                            'chunk_type': 'claim_independent',
                            'chunk_index': idx,
                            'text': claim_clean,
                            'tokens_approx': len(claim_clean) // 4
                        })
            
            # Dependent claims (grouped)
            # Group dependent claims into chunks of ~300 tokens
            if parsed_claims['dependent']:
                current_group = []
                current_tokens = 0
                group_idx = 0
                
                for claim in parsed_claims['dependent']:
                    claim_clean = self.clean_text(claim)
                    claim_tokens = len(claim_clean) // 4
                    
                    if current_tokens + claim_tokens > 300 and current_group:
                        # Save current group
                        chunks.append({
                            'patent_url': patent_id,
                            'title': title,
                            'publication_date': pub_date,
                            'chunk_type': 'claim_dependent',
                            'chunk_index': group_idx,
                            'text': ' '.join(current_group),
                            'tokens_approx': current_tokens
                        })
                        current_group = [claim_clean]
                        current_tokens = claim_tokens
                        group_idx += 1
                    else:
                        current_group.append(claim_clean)
                        current_tokens += claim_tokens
                
                # Add remaining group
                if current_group:
                    chunks.append({
                        'patent_url': patent_id,
                        'title': title,
                        'publication_date': pub_date,
                        'chunk_type': 'claim_dependent',
                        'chunk_index': group_idx,
                        'text': ' '.join(current_group),
                        'tokens_approx': current_tokens
                    })
        
        # 3. Description chunks (semantic paragraph level)
        if patent.get('description'):
            description_clean = self.clean_text(patent['description'])
            desc_chunks = self.semantic_split(description_clean, max_tokens=300)
            
            for idx, desc_chunk in enumerate(desc_chunks):
                if desc_chunk:
                    chunks.append({
                        'patent_url': patent_id,
                        'title': title,
                        'publication_date': pub_date,
                        'chunk_type': 'description',
                        'chunk_index': idx,
                        'text': desc_chunk,
                        'tokens_approx': len(desc_chunk) // 4
                    })
        
        return chunks
    
    def normalize_all(self) -> List[Dict]:
        """
        Process all patents and create normalized chunks.
        
        Returns:
            List of all chunks from all patents
        """
        logger.info(f"Starting normalization of {len(self.patents)} patents...")
        
        all_chunks = []
        for idx, patent in enumerate(self.patents, 1):
            logger.info(f"Processing patent {idx}/{len(self.patents)}: {patent.get('title', 'Unknown')[:50]}")
            chunks = self.chunk_patent(patent)
            all_chunks.extend(chunks)
        
        self.normalized_chunks = all_chunks
        logger.info(f"Normalization complete. Created {len(all_chunks)} chunks from {len(self.patents)} patents")
        
        # Print statistics
        chunk_types = {}
        for chunk in all_chunks:
            chunk_type = chunk['chunk_type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        logger.info("Chunk distribution:")
        for chunk_type, count in sorted(chunk_types.items()):
            logger.info(f"  {chunk_type}: {count} chunks")
        
        return all_chunks
    
    def save_normalized(self, output_path: str = 'normalized_patents.json'):
        """
        Save normalized chunks to JSON file.
        
        Args:
            output_path: Path for the output JSON file
        """
        if not self.normalized_chunks:
            logger.warning("No normalized chunks to save")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.normalized_chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Normalized chunks saved to {output_path}")
        
        # Calculate and display statistics
        total_tokens = sum(chunk['tokens_approx'] for chunk in self.normalized_chunks)
        avg_tokens = total_tokens / len(self.normalized_chunks) if self.normalized_chunks else 0
        
        logger.info(f"Total chunks: {len(self.normalized_chunks)}")
        logger.info(f"Total tokens (approx): {total_tokens}")
        logger.info(f"Average tokens per chunk: {avg_tokens:.1f}")


def main():
    """Example usage of the PatentNormalizer."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python normalize.py <input_json_path> [output_json_path]")
        print("Example: python normalize.py scraped_patents.json normalized_patents.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'normalized_patents.json'
    
    # Create normalizer instance
    normalizer = PatentNormalizer(input_path)
    
    # Load patents
    if not normalizer.load_patents():
        sys.exit(1)
    
    # Normalize all patents
    normalizer.normalize_all()
    
    # Save results
    normalizer.save_normalized(output_path)
    
    print("\n" + "="*50)
    print("NORMALIZATION COMPLETE")
    print("="*50)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total chunks: {len(normalizer.normalized_chunks)}")


if __name__ == "__main__":
    main()
