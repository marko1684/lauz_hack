#!/usr/bin/env python3
"""
Fix full_embeddings.pkl to add missing metadata fields
"""

import pickle
import sys

def fix_embeddings_file(input_path, output_path=None):
    """Add missing metadata to embeddings file"""
    
    if output_path is None:
        output_path = input_path
    
    print(f"Loading embeddings from: {input_path}")
    
    # Load existing data
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Current keys: {data.keys()}")
    
    # Add missing fields if not present
    if 'model_type' not in data:
        data['model_type'] = 'sentence-transformers'
        print("✅ Added model_type: sentence-transformers")
    
    if 'embedding_dim' not in data:
        data['embedding_dim'] = data['embeddings'].shape[1]
        print(f"✅ Added embedding_dim: {data['embedding_dim']}")
    
    # Save updated data
    print(f"Saving fixed embeddings to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("✅ Done!")
    print(f"Total chunks: {len(data['chunks'])}")
    print(f"Embeddings shape: {data['embeddings'].shape}")
    print(f"Model type: {data['model_type']}")
    print(f"Embedding dimension: {data['embedding_dim']}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_embeddings.py <embeddings_file.pkl> [output_file.pkl]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file
    
    fix_embeddings_file(input_file, output_file)
