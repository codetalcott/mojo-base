# CLAUDE.md - Mojo Semantic Search Project

This file provides guidance to Claude Code (claude.ai/code) when working with the Mojo semantic search system.

## Project Overview

Real-time cross-project semantic code search powered by custom Mojo kernels and onedev portfolio intelligence.

## Essential Resources

### LLM Context Files
- `llms.txt` - Quick reference guide for LLM agents
- `llms-full.txt` - Comprehensive context including onedev integration
- `llms-mojo.txt` - Mojo-specific implementation patterns
- `llms-python.txt` - Python integration patterns
- `llm-onedev.txt` - Guide to onedev MCP tools integration

## Quick Start

```bash
# Activate Mojo environment
cd portfolio-search
pixi run mojo ../semantic_search_mvp.mojo
```

## Onedev MCP Tools

The project uses onedev's 69 MCP tools across 9 domains. See `llm-onedev.txt` for integration guide.

Key tools for semantic search:
- `search_codebase_knowledge` - Semantic search across codebase
- `assemble_context` - AI context generation with embeddings
- `find_patterns` - Pattern detection and matching
- `get_vector_similarity_insights` - Vector similarity analysis

## Development Focus

When working on this project, prioritize:
1. High-performance Mojo kernel optimization
2. Real-time semantic search (< 50ms latency)
3. Cross-project pattern detection
4. Onedev portfolio intelligence integration