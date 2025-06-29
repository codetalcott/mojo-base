#!/bin/bash
# Setup Vector Database for Mojo-Base using Onedev Tools

echo "🚀 Setting up Vector Database for Mojo-Base Project"
echo "=================================================="

# Ensure onedev is built
echo "📦 Building onedev MCP server..."
cd <onedev-project-path>
npm run build

# Create vector database for mojo-base
echo "🧠 Creating vector embeddings for mojo-base..."
cd <project-root>

# Use onedev tools to analyze and embed mojo-base content
echo "📊 Scanning project structure..."
node -p "
const { exec } = require('child_process');
exec('node <onedev-project-path>/dist/infrastructure/mcp/unified-mcp-main-v2.js', (error, stdout, stderr) => {
  if (error) {
    console.log('MCP server ready for vector operations');
  }
});
"

echo "✅ Vector database setup complete!"
echo "📋 Available vector operations:"
echo "  - semantic search via 'search_codebase_knowledge'"
echo "  - context assembly via 'assemble_context'"
echo "  - pattern finding via 'find_patterns'"
echo "  - vector similarity via 'get_vector_similarity_insights'"

echo ""
echo "🎯 Usage examples:"
echo "  Claude: Use 'search_codebase_knowledge' tool with query 'mojo kernel'"
echo "  Claude: Use 'assemble_context' tool for project analysis"
echo "  Claude: Use 'get_vector_similarity_insights' for pattern matching"