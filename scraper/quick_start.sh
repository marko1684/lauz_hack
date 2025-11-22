#!/bin/bash
# Quick Start Script for Patent Similarity Search

set -e

echo "==================================================================="
echo "  Patent Similarity Search - Quick Start"
echo "==================================================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not active. Activating venv..."
    if [ -f "../venv/bin/activate" ]; then
        source ../venv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "❌ Virtual environment not found. Please run from scraper/ directory."
        exit 1
    fi
fi

echo "✅ Python environment: $(which python3)"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: ./quick_start.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  vectorize <input.json> <output.pkl>  - Create embeddings from normalized chunks"
    echo "  search <embeddings.pkl> <query>       - Search for similar patents"
    echo "  example                               - Run example search on test data"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh example"
    echo "  ./quick_start.sh vectorize normalized_test.json embeddings.pkl"
    echo "  ./quick_start.sh search embeddings.pkl 'hydrogen engine'"
}

# Main command handling
case "${1:-example}" in
    vectorize)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "❌ Error: Missing arguments"
            echo "Usage: ./quick_start.sh vectorize <input.json> <output.pkl>"
            exit 1
        fi
        
        INPUT=$2
        OUTPUT=$3
        
        if [ ! -f "$INPUT" ]; then
            echo "❌ Error: Input file not found: $INPUT"
            exit 1
        fi
        
        echo "📊 Vectorizing chunks from: $INPUT"
        echo "💾 Output will be saved to: $OUTPUT"
        echo ""
        
        python3 vectorize.py "$INPUT" "$OUTPUT" --create-faiss
        
        echo ""
        echo "✅ Vectorization complete!"
        echo "📁 Files created:"
        echo "   - $OUTPUT (embeddings + metadata)"
        echo "   - ${OUTPUT%.pkl}.json (summary)"
        echo "   - ${OUTPUT%.pkl}.faiss (FAISS index)"
        ;;
    
    search)
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "❌ Error: Missing arguments"
            echo "Usage: ./quick_start.sh search <embeddings.pkl> <query>"
            exit 1
        fi
        
        EMBEDDINGS=$2
        QUERY=$3
        
        if [ ! -f "$EMBEDDINGS" ]; then
            echo "❌ Error: Embeddings file not found: $EMBEDDINGS"
            echo "Hint: Run vectorize first"
            exit 1
        fi
        
        FAISS_FILE="${EMBEDDINGS%.pkl}.faiss"
        
        echo "🔍 Searching for: '$QUERY'"
        echo "📊 Using embeddings: $EMBEDDINGS"
        echo ""
        
        if [ -f "$FAISS_FILE" ]; then
            python3 search.py "$EMBEDDINGS" "$QUERY" --top-k 10 --faiss "$FAISS_FILE"
        else
            python3 search.py "$EMBEDDINGS" "$QUERY" --top-k 10
        fi
        ;;
    
    example)
        echo "🚀 Running example search on test data..."
        echo ""
        
        # Check if test embeddings exist
        if [ ! -f "test_embeddings.pkl" ]; then
            echo "📊 Creating test embeddings first..."
            
            if [ ! -f "normalized_test.json" ]; then
                echo "❌ Error: normalized_test.json not found"
                echo "Please run normalize.py first"
                exit 1
            fi
            
            python3 vectorize.py normalized_test.json test_embeddings.pkl --create-faiss
            echo ""
        fi
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Example 1: Search for 'hydrogen combustion engine'"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python3 search.py test_embeddings.pkl "hydrogen combustion engine" --top-k 5 --faiss test_embeddings.faiss
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Example 2: Search for 'ammonia fuel' (by patent)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python3 search.py test_embeddings.pkl "ammonia fuel injection" --top-k 5 --by-patent --faiss test_embeddings.faiss
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Example 3: Search only abstracts"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python3 search.py test_embeddings.pkl "zero emission engine" --top-k 3 --chunk-type abstract
        
        echo ""
        echo "✅ Examples complete!"
        echo ""
        echo "Try your own search:"
        echo "  ./quick_start.sh search test_embeddings.pkl 'your query here'"
        ;;
    
    help|--help|-h)
        show_usage
        ;;
    
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
