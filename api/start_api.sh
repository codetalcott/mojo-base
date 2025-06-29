#!/bin/bash
# Start Mojo Semantic Search API
# Real portfolio corpus with MCP integration

echo "🚀 Starting Mojo Semantic Search API"
echo "===================================="
echo "Real portfolio corpus: 2,637 vectors from 44 projects"
echo "MCP portfolio intelligence: Enabled"
echo "Performance optimization: 128-dim vectors (6x boost)"
echo ""

# Set working directory
cd "$(dirname "$0")/.."

# Check if corpus exists
if [ ! -f "data/portfolio_corpus.json" ]; then
    echo "❌ Error: Portfolio corpus not found"
    echo "Run: python3 src/corpus/portfolio_corpus_builder.py"
    exit 1
fi

# Check Python dependencies
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "📦 Installing API dependencies..."
    pip install fastapi uvicorn requests pydantic
fi

echo "✅ Dependencies ready"
echo "✅ Portfolio corpus: $(python3 -c "import json; print(json.load(open('data/portfolio_corpus.json'))['metadata']['total_vectors'])" 2>/dev/null || echo "Unknown") vectors"
echo ""

echo "🌐 Starting API server..."
echo "  URL: http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo "  Health: http://localhost:8000/health"
echo ""

# Start the API server
python3 api/semantic_search_api.py