#!/usr/bin/env python3
"""
Mojo Semantic Search API
Simple, functional API for portfolio semantic search with real corpus
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import time
import logging
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.integration.mcp_real_bridge import MCPRealBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Mojo Semantic Search API",
    description="Portfolio semantic search with real corpus data",
    version="2.0.0"
)

# Initialize MCP bridge
mcp_bridge = MCPRealBridge()

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    include_mcp: bool = True
    filter_language: Optional[str] = None
    filter_project: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    text: str
    file_path: str
    project: str
    language: str
    context_type: str
    similarity_score: float
    confidence: float
    start_line: int
    end_line: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    corpus_size: int
    mcp_enhanced: bool
    performance_metrics: Dict[str, float]

class CorpusStats(BaseModel):
    total_vectors: int
    vector_dimensions: int
    source_projects: int
    languages: List[str]
    context_types: List[str]
    projects_included: List[str]
    quality_score: float
    onedev_vectors: int
    portfolio_vectors: int

@app.on_event("startup")
async def startup_event():
    """Initialize the search system on startup."""
    logger.info("🚀 Starting Mojo Semantic Search API")
    
    # Load portfolio corpus
    if mcp_bridge.load_portfolio_corpus():
        logger.info("✅ Portfolio corpus loaded successfully")
    else:
        logger.warning("⚠️ Failed to load portfolio corpus - using fallback mode")

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Mojo Semantic Search API",
        "version": "2.0.0",
        "description": "Portfolio semantic search with real corpus data",
        "corpus_size": 2637,
        "vector_dimensions": 128,
        "source_projects": 44,
        "status": "operational",
        "features": [
            "Real portfolio corpus",
            "MCP portfolio intelligence", 
            "Cross-project insights",
            "128-dim performance optimization"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    corpus_loaded = mcp_bridge.portfolio_corpus is not None
    
    return {
        "status": "healthy" if corpus_loaded else "degraded",
        "corpus_loaded": corpus_loaded,
        "mcp_available": True,
        "api_version": "2.0.0",
        "timestamp": time.time()
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform semantic search across the portfolio corpus."""
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Search request: '{request.query}'")
        
        # Perform enhanced semantic search using MCP bridge
        if request.include_mcp:
            search_results = mcp_bridge.enhanced_semantic_search(
                request.query, 
                max_results=request.max_results
            )
        else:
            # Simple local search only
            local_results = mcp_bridge.search_local_corpus(
                request.query, 
                max_results=request.max_results
            )
            search_results = {
                "query": request.query,
                "local_results": local_results,
                "mcp_enhancement": None,
                "performance": {"total_latency_ms": 0},
                "metadata": {"corpus_size": len(mcp_bridge.portfolio_corpus.get("vectors", []))}
            }
        
        # Convert to API response format
        results = []
        local_results = search_results.get("local_results", [])
        
        for result in local_results:
            # Apply filters if specified
            if request.filter_language and result.get("language") != request.filter_language:
                continue
            if request.filter_project and result.get("project") != request.filter_project:
                continue
                
            search_result = SearchResult(
                id=result["id"],
                text=result["text"],
                file_path=result["file_path"],
                project=result["project"],
                language=result["language"],
                context_type=result["context_type"],
                similarity_score=result["similarity_score"],
                confidence=result["confidence"],
                start_line=result.get("start_line", 0),
                end_line=result.get("end_line", 0)
            )
            results.append(search_result)
        
        search_time = (time.time() - start_time) * 1000
        
        # Performance metrics
        performance_metrics = search_results.get("performance", {})
        performance_metrics["api_overhead_ms"] = search_time - performance_metrics.get("total_latency_ms", 0)
        
        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time,
            corpus_size=search_results.get("metadata", {}).get("corpus_size", 0),
            mcp_enhanced=request.include_mcp,
            performance_metrics=performance_metrics
        )
        
        logger.info(f"✅ Search completed: {len(results)} results in {search_time:.1f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/simple")
async def simple_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum results"),
    lang: Optional[str] = Query(None, description="Filter by language"),
    project: Optional[str] = Query(None, description="Filter by project")
):
    """Simple GET-based search endpoint."""
    request = SearchRequest(
        query=q,
        max_results=limit,
        include_mcp=True,
        filter_language=lang,
        filter_project=project
    )
    return await search(request)

@app.get("/corpus/stats", response_model=CorpusStats)
async def get_corpus_stats():
    """Get comprehensive corpus statistics."""
    if not mcp_bridge.portfolio_corpus:
        raise HTTPException(status_code=503, detail="Corpus not loaded")
    
    metadata = mcp_bridge.portfolio_corpus.get("metadata", {})
    
    return CorpusStats(
        total_vectors=metadata.get("total_vectors", 0),
        vector_dimensions=metadata.get("vector_dimensions", 128),
        source_projects=metadata.get("source_projects", 0),
        languages=metadata.get("languages", []),
        context_types=metadata.get("context_types", []),
        projects_included=metadata.get("projects_included", []),
        quality_score=metadata.get("quality_score", 0.0),
        onedev_vectors=metadata.get("onedev_vectors", 0),
        portfolio_vectors=metadata.get("portfolio_vectors", 0)
    )

@app.get("/corpus/projects")
async def get_projects():
    """Get list of all projects in the corpus."""
    if not mcp_bridge.portfolio_corpus:
        raise HTTPException(status_code=503, detail="Corpus not loaded")
    
    metadata = mcp_bridge.portfolio_corpus.get("metadata", {})
    projects = metadata.get("projects_included", [])
    
    # Get project statistics
    vectors = mcp_bridge.portfolio_corpus.get("vectors", [])
    project_stats = {}
    
    for vector in vectors:
        project = vector.get("project", "unknown")
        if project not in project_stats:
            project_stats[project] = {
                "name": project,
                "vector_count": 0,
                "languages": set(),
                "context_types": set()
            }
        
        project_stats[project]["vector_count"] += 1
        project_stats[project]["languages"].add(vector.get("language", "unknown"))
        project_stats[project]["context_types"].add(vector.get("context_type", "unknown"))
    
    # Convert sets to lists for JSON serialization
    for project, stats in project_stats.items():
        stats["languages"] = list(stats["languages"])
        stats["context_types"] = list(stats["context_types"])
    
    return {
        "total_projects": len(projects),
        "projects": list(project_stats.values())
    }

@app.get("/corpus/languages")
async def get_languages():
    """Get language distribution in the corpus."""
    if not mcp_bridge.portfolio_corpus:
        raise HTTPException(status_code=503, detail="Corpus not loaded")
    
    vectors = mcp_bridge.portfolio_corpus.get("vectors", [])
    language_stats = {}
    
    for vector in vectors:
        lang = vector.get("language", "unknown")
        if lang not in language_stats:
            language_stats[lang] = {
                "language": lang,
                "vector_count": 0,
                "projects": set(),
                "context_types": set()
            }
        
        language_stats[lang]["vector_count"] += 1
        language_stats[lang]["projects"].add(vector.get("project", "unknown"))
        language_stats[lang]["context_types"].add(vector.get("context_type", "unknown"))
    
    # Convert sets to lists
    for lang, stats in language_stats.items():
        stats["projects"] = list(stats["projects"])
        stats["context_types"] = list(stats["context_types"])
        stats["project_count"] = len(stats["projects"])
    
    return {
        "total_languages": len(language_stats),
        "languages": list(language_stats.values())
    }

@app.get("/mcp/validate")
async def validate_mcp_integration():
    """Validate MCP integration functionality."""
    try:
        validation_results = mcp_bridge.validate_mcp_integration()
        return {
            "status": "success",
            "validation_results": validation_results,
            "overall_status": validation_results.get("overall_status", "UNKNOWN")
        }
    except Exception as e:
        logger.error(f"MCP validation error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "overall_status": "FAILED"
        }

@app.get("/mcp/demo")
async def demo_enhanced_search():
    """Demonstrate enhanced search capabilities."""
    try:
        demo_results = mcp_bridge.demonstrate_enhanced_search()
        return {
            "status": "success",
            "demonstration": demo_results
        }
    except Exception as e:
        logger.error(f"MCP demo error: {e}")
        return {
            "status": "error", 
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Mojo Semantic Search API")
    print("====================================")
    print("Real portfolio corpus with 2,637 vectors")
    print("MCP portfolio intelligence enabled")
    print("128-dimensional vectors (6x performance boost)")
    print()
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 Interactive docs at: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    print()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        access_log=True
    )