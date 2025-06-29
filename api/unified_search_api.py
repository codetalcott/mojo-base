#!/usr/bin/env python3
"""
Unified Search API
Combines semantic search with incremental corpus management
Single API endpoint for all search and corpus operations
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.search.semantic_search_engine import SemanticSearchEngine
from src.corpus.incremental_updater import IncrementalUpdater

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    include_mcp: bool = True
    filter_project: Optional[str] = None
    filter_language: Optional[str] = None

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
    performance_metrics: Dict[str, Any]

class UpdateRequest(BaseModel):
    file_path: str
    content: str
    project: str
    language: str

class ProjectRequest(BaseModel):
    name: str
    path: str

class UpdateResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Initialize FastAPI app
app = FastAPI(
    title="Mojo Semantic Search - Unified API",
    description="Combined semantic search and corpus management API",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
search_engine = None
corpus_updater = None

@app.on_event("startup")
async def startup_event():
    """Initialize search engine and corpus updater on startup."""
    global search_engine, corpus_updater
    
    print("🚀 Initializing Unified Search API...")
    
    # Initialize components
    corpus_updater = IncrementalUpdater()
    search_engine = SemanticSearchEngine()
    
    # Load corpus into search engine
    if corpus_updater.vectors is not None:
        search_engine.vectors = corpus_updater.vectors
        search_engine.metadata = corpus_updater.metadata
        print(f"✅ Loaded {len(corpus_updater.metadata)} vectors into search engine")
    else:
        print("⚠️  No corpus found - search will not work until corpus is built")

@app.get("/")
async def root():
    """Root endpoint."""
    stats = corpus_updater.get_corpus_stats() if corpus_updater else {}
    return {
        "service": "Mojo Semantic Search - Unified API",
        "version": "2.0.0",
        "status": "running",
        "corpus_vectors": stats.get('total_vectors', 0),
        "search_ready": search_engine is not None and len(search_engine.metadata) > 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    corpus_ready = corpus_updater and corpus_updater.vectors is not None
    search_ready = search_engine and len(search_engine.metadata) > 0
    
    return {
        "status": "healthy",
        "corpus_loaded": corpus_ready,
        "search_ready": search_ready,
        "total_vectors": len(search_engine.metadata) if search_ready else 0
    }

# ====== SEARCH ENDPOINTS ======

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform semantic search."""
    if not search_engine or len(search_engine.metadata) == 0:
        raise HTTPException(status_code=503, detail="Search engine not ready - no corpus loaded")
    
    try:
        results = await search_engine.search(
            query=request.query,
            max_results=request.max_results,
            include_mcp=request.include_mcp
        )
        
        # Apply filters
        filtered_results = results['results']
        
        if request.filter_project:
            filtered_results = [r for r in filtered_results if r.get('project') == request.filter_project]
            
        if request.filter_language:
            filtered_results = [r for r in filtered_results if r.get('language') == request.filter_language]
        
        # Convert to response format
        search_results = [
            SearchResult(
                id=result.get('id', ''),
                text=result.get('text', ''),
                file_path=result.get('file_path', ''),
                project=result.get('project', ''),
                language=result.get('language', ''),
                context_type=result.get('context_type', ''),
                similarity_score=result.get('similarity_score', 0.0),
                confidence=result.get('confidence', 0.0),
                start_line=result.get('start_line', 0),
                end_line=result.get('end_line', 0)
            )
            for result in filtered_results[:request.max_results]
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=results.get('search_time_ms', 0),
            performance_metrics=results.get('performance_metrics', {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search/simple")
async def simple_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum results"),
    project: Optional[str] = Query(None, description="Filter by project"),
    lang: Optional[str] = Query(None, description="Filter by language")
):
    """Simple search endpoint compatible with existing web interface."""
    request = SearchRequest(
        query=q,
        max_results=limit,
        filter_project=project,
        filter_language=lang
    )
    return await search(request)

# ====== CORPUS MANAGEMENT ENDPOINTS ======

@app.get("/corpus/stats")
async def get_corpus_stats():
    """Get corpus statistics."""
    if not corpus_updater:
        raise HTTPException(status_code=503, detail="Corpus updater not initialized")
    
    return corpus_updater.get_corpus_stats()

@app.post("/corpus/add-file", response_model=UpdateResponse)
async def add_file_to_corpus(request: UpdateRequest):
    """Add a single file to the corpus."""
    if not corpus_updater:
        raise HTTPException(status_code=503, detail="Corpus updater not initialized")
    
    try:
        chunks = await corpus_updater.add_file_to_corpus(
            request.file_path,
            request.content,
            request.project,
            request.language
        )
        
        corpus_updater.save_corpus()
        
        # Update search engine
        if search_engine:
            search_engine.vectors = corpus_updater.vectors
            search_engine.metadata = corpus_updater.metadata
        
        return UpdateResponse(
            success=True,
            message=f"Added {len(chunks)} chunks from {request.file_path}",
            data={
                "chunks_added": len(chunks),
                "file_path": request.file_path,
                "project": request.project
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add file: {str(e)}")

@app.post("/corpus/upload-file", response_model=UpdateResponse)
async def upload_file_to_corpus(
    file: UploadFile = File(...),
    project: str = "uploaded",
    language: Optional[str] = None
):
    """Upload and add a file to the corpus."""
    if not corpus_updater:
        raise HTTPException(status_code=503, detail="Corpus updater not initialized")
    
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        if not language:
            file_ext = Path(file.filename).suffix
            language = corpus_updater.corpus_builder.detect_language(file_ext)
        
        chunks = await corpus_updater.add_file_to_corpus(
            file.filename,
            text_content,
            project,
            language
        )
        
        corpus_updater.save_corpus()
        
        # Update search engine
        if search_engine:
            search_engine.vectors = corpus_updater.vectors
            search_engine.metadata = corpus_updater.metadata
        
        return UpdateResponse(
            success=True,
            message=f"Uploaded and added {len(chunks)} chunks from {file.filename}",
            data={
                "chunks_added": len(chunks),
                "file_path": file.filename,
                "project": project,
                "language": language
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/corpus/add-project", response_model=UpdateResponse)
async def add_project_to_corpus(request: ProjectRequest):
    """Add an entire project to the corpus."""
    if not corpus_updater:
        raise HTTPException(status_code=503, detail="Corpus updater not initialized")
    
    try:
        updates = await corpus_updater.scan_project_for_updates(request.path, request.name)
        
        # Update search engine
        if search_engine:
            search_engine.vectors = corpus_updater.vectors
            search_engine.metadata = corpus_updater.metadata
        
        return UpdateResponse(
            success=True,
            message=f"Added project {request.name}",
            data={
                "project_name": request.name,
                "files_added": len(updates['added']),
                "total_chunks": sum(update['chunks_added'] for update in updates['added']),
                "errors": updates['errors']
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add project: {str(e)}")

@app.delete("/corpus/remove-project/{project_name}", response_model=UpdateResponse)
async def remove_project_from_corpus(project_name: str):
    """Remove all files from a project from the corpus."""
    if not corpus_updater:
        raise HTTPException(status_code=503, detail="Corpus updater not initialized")
    
    try:
        import sqlite3
        removed_count = 0
        
        with sqlite3.connect(corpus_updater.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT file_path FROM file_hashes WHERE project = ?",
                (project_name,)
            )
            file_paths = [row[0] for row in cursor.fetchall()]
            
            for file_path in file_paths:
                corpus_updater.remove_file_chunks(file_path)
                removed_count += 1
                
        corpus_updater.save_corpus()
        
        # Update search engine
        if search_engine:
            search_engine.vectors = corpus_updater.vectors
            search_engine.metadata = corpus_updater.metadata
        
        return UpdateResponse(
            success=True,
            message=f"Removed {removed_count} files from project {project_name}",
            data={
                "project_name": project_name,
                "files_removed": removed_count
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove project: {str(e)}")

@app.get("/corpus/projects")
async def list_projects():
    """List all projects in the corpus."""
    if not corpus_updater:
        raise HTTPException(status_code=503, detail="Corpus updater not initialized")
    
    stats = corpus_updater.get_corpus_stats()
    return {
        "projects": stats['projects'],
        "total_projects": len(stats['projects'])
    }

@app.get("/corpus/files/{project_name}")
async def get_project_files(project_name: str):
    """Get all files for a specific project."""
    if not corpus_updater:
        raise HTTPException(status_code=503, detail="Corpus updater not initialized")
    
    try:
        import sqlite3
        
        with sqlite3.connect(corpus_updater.db_path) as conn:
            cursor = conn.execute("""
                SELECT file_path, language, chunk_count, last_updated
                FROM file_hashes 
                WHERE project = ?
                ORDER BY last_updated DESC
            """, (project_name,))
            
            files = [
                {
                    "file_path": row[0],
                    "language": row[1], 
                    "chunk_count": row[2],
                    "last_updated": row[3]
                }
                for row in cursor.fetchall()
            ]
            
        return {
            "project": project_name,
            "files": files,
            "total_files": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project files: {str(e)}")

# ====== PERFORMANCE ENDPOINTS ======

@app.get("/performance/validate")
async def validate_performance():
    """Validate system performance."""
    if not search_engine or len(search_engine.metadata) == 0:
        raise HTTPException(status_code=503, detail="Search engine not ready")
    
    try:
        # Run a test search
        test_query = "authentication patterns"
        results = await search_engine.search(query=test_query, max_results=5)
        
        return {
            "status": "performance_validation_complete",
            "test_query": test_query,
            "results_found": len(results['results']),
            "search_time_ms": results.get('search_time_ms', 0),
            "performance_metrics": results.get('performance_metrics', {}),
            "corpus_size": len(search_engine.metadata)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance validation failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Unified Search API...")
    print("=" * 50)
    print("🔍 Semantic search endpoints:")
    print("   POST /search - Advanced search")
    print("   GET  /search/simple - Simple search")
    print()
    print("📊 Corpus management endpoints:")
    print("   GET  /corpus/stats - Corpus statistics")
    print("   POST /corpus/add-file - Add single file")
    print("   POST /corpus/upload-file - Upload file")
    print("   POST /corpus/add-project - Add project")
    print()
    print("🌐 Web interface: http://localhost:8000/docs")
    print("🔧 Health check: http://localhost:8000/health")
    print()
    
    uvicorn.run(
        "unified_search_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )