# Mojo Semantic Search - Portfolio Intelligence

Real-time cross-project semantic code search powered by MAX Graph API and custom Mojo kernels.

## Quick Start

```bash
# Web Interface (Recommended)
python api/semantic_search_api_v2.py  # Start API server
python tests/web/start_web_demo.py    # Start web interface

# Direct Mojo Search
cd portfolio-search
pixi run mojo ../semantic_search_mvp.mojo

# Test Web Interface
python tests/web/test_web_status.py
```

## Performance

- **MAX Graph**: 1.8ms search latency (7x faster)
- **Legacy Mojo**: 12.7ms search latency 
- **Corpus**: 2,637 real code snippets from 44 projects
- **Throughput**: 10-50x faster than traditional search

## Architecture

### MAX Graph Implementation
- GPU-optimized semantic search with automatic kernel fusion
- Hardware-agnostic execution with memory optimization
- Real-time compilation for target devices

### Legacy Mojo Kernels  
- Custom MLA (Multi-Head Latent Attention) kernels
- BMM (Batched Matrix Multiplication) for similarity search
- SIMD-accelerated vector operations

### Onedev Integration
- Portfolio intelligence with 69 MCP tools across 9 domains
- Cross-project pattern detection and insights
- Graceful fallback when onedev unavailable

## Directory Structure

```
tests/               # All test files organized by category
├── max_graph_debug/ # MAX Graph API debugging and validation
├── performance/     # Performance benchmarking and validation
└── web/            # Web interface testing and demos

docs/               # Documentation organized by type
├── reports/        # Status reports and project summaries
└── status-reports/ # Development progress tracking

data/
├── results/        # Performance and benchmark results
└── portfolio_corpus.json # Main vector database

src/
├── max_graph/      # MAX Graph API implementation
├── kernels/        # Legacy Mojo kernel implementations
├── search/         # Core search engine logic
└── integration/    # Onedev MCP bridge and corpus loading
```

## Key Files

- `semantic_search_mvp.mojo` - Main semantic search implementation
- `src/max_graph/semantic_search_graph.py` - MAX Graph integration
- `api/semantic_search_api_v2.py` - Production API server
- `web/index.html` - Interactive web interface with real-time search
- `tests/web/test_web_status.py` - Complete web interface validation

## Usage Examples

### Web Interface
Access at `http://localhost:8080` after starting both API and web servers.
Try queries like:
- "authentication patterns" 
- "React components"
- "API error handling"

### Direct API
```bash
curl "http://localhost:8000/search/simple?q=authentication&limit=5"
```

### Mojo Integration
```mojo
# Import and use in your Mojo code
from semantic_search_mvp import search_portfolio
let results = search_portfolio("database connections", 10)
```

## Performance Benchmarks

| Implementation | Latency | Throughput | Use Case |
|---------------|---------|------------|----------|
| MAX Graph | 1.8ms | 500k vectors/sec | Production |
| Legacy Mojo | 12.7ms | 75k vectors/sec | Fallback |
| Traditional | 100-500ms | 2k vectors/sec | Baseline |

**Status**: Production ready with working web interface and comprehensive test coverage.