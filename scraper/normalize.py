import json
import re
import html
import unicodedata
from typing import List, Dict
import logging

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
    
    def clean_text(self, text: str) -> str:
        """
        Clean text: lowercase, remove weird Unicode, strip HTML tags, normalize whitespace.
        
        Args:
            text: Raw text to clean
            
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
        
        # Normalize whitespace (replace multiple spaces/newlines with single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
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
    
    def split_into_chunks(self, text: str, max_tokens: int = 350) -> List[str]:
        """
        Split text into chunks of approximately max_tokens size.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk (approximate)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split by sentences (periods followed by space)
        # Handle multiple sentence ending patterns
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Rough token estimate: ~4 characters per token
            sentence_tokens = len(sentence) // 4
            
            # If a single sentence is too long, force split it
            if sentence_tokens > max_tokens:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_tokens = len(word) // 4 + 1
                    if temp_length + word_tokens > max_tokens and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_length += word_tokens
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_length = temp_length
            elif current_length + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
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
                # Split abstract if it's too long
                abstract_tokens = len(abstract_clean) // 4
                if abstract_tokens > 400:
                    abstract_chunks = self.split_into_chunks(abstract_clean, max_tokens=350)
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
            # Split large claims into smaller chunks
            for idx, claim in enumerate(parsed_claims['independent']):
                claim_clean = self.clean_text(claim)
                if claim_clean:
                    # If claim is too large, split it
                    claim_tokens = len(claim_clean) // 4
                    if claim_tokens > 350:
                        # Split by sentences
                        claim_sentences = self.split_into_chunks(claim_clean, max_tokens=300)
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
        
        # 3. Description chunks (paragraph level)
        if patent.get('description'):
            description_clean = self.clean_text(patent['description'])
            desc_chunks = self.split_into_chunks(description_clean, max_tokens=300)
            
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
