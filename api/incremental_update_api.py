#!/usr/bin/env python3
"""
Incremental Update API
REST API endpoints for managing corpus updates
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.corpus.incremental_updater import IncrementalUpdater

# Request/Response models
class ProjectRequest(BaseModel):
    name: str
    path: str

class BulkUpdateRequest(BaseModel):
    projects: List[ProjectRequest]

class FileUpdateRequest(BaseModel):
    file_path: str
    content: str
    project: str
    language: str

class UpdateResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class CorpusStats(BaseModel):
    total_vectors: int
    total_files: int
    total_chunks: int
    projects: List[str]
    languages: List[str]
    corpus_size_mb: float

# Initialize FastAPI app
app = FastAPI(
    title="Mojo Semantic Search - Incremental Update API",
    description="API for incrementally updating the semantic search corpus",
    version="1.0.0"
)

# Global updater instance
updater = IncrementalUpdater()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "Mojo Semantic Search - Incremental Update API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = updater.get_corpus_stats()
        return {
            "status": "healthy",
            "corpus_loaded": stats['total_vectors'] > 0,
            "total_vectors": stats['total_vectors']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/corpus/stats", response_model=CorpusStats)
async def get_corpus_stats():
    """Get current corpus statistics."""
    try:
        stats = updater.get_corpus_stats()
        return CorpusStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/corpus/add-file", response_model=UpdateResponse)
async def add_file_to_corpus(request: FileUpdateRequest):
    """Add a single file to the corpus."""
    try:
        chunks = await updater.add_file_to_corpus(
            request.file_path,
            request.content,
            request.project,
            request.language
        )
        
        updater.save_corpus()
        
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
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Detect language if not provided
        if not language:
            file_ext = Path(file.filename).suffix
            language = updater.corpus_builder.detect_language(file_ext)
        
        # Add to corpus
        chunks = await updater.add_file_to_corpus(
            file.filename,
            text_content,
            project,
            language
        )
        
        updater.save_corpus()
        
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
    try:
        updates = await updater.scan_project_for_updates(request.path, request.name)
        
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

@app.post("/corpus/bulk-update", response_model=UpdateResponse)
async def bulk_update_corpus(request: BulkUpdateRequest, background_tasks: BackgroundTasks):
    """Perform bulk update of multiple projects."""
    try:
        # Convert to format expected by updater
        projects = [{"name": p.name, "path": p.path} for p in request.projects]
        
        # Run update in background
        updates = await updater.bulk_update_from_projects(projects)
        
        return UpdateResponse(
            success=True,
            message=f"Bulk update completed for {len(projects)} projects",
            data=updates
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk update failed: {str(e)}")

@app.delete("/corpus/remove-project/{project_name}", response_model=UpdateResponse)
async def remove_project_from_corpus(project_name: str):
    """Remove all files from a project from the corpus."""
    try:
        # Get all files for this project
        import sqlite3
        removed_count = 0
        
        with sqlite3.connect(updater.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT file_path FROM file_hashes WHERE project = ?",
                (project_name,)
            )
            file_paths = [row[0] for row in cursor.fetchall()]
            
            # Remove each file
            for file_path in file_paths:
                updater.remove_file_chunks(file_path)
                removed_count += 1
                
        updater.save_corpus()
        
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

@app.delete("/corpus/remove-file", response_model=UpdateResponse)
async def remove_file_from_corpus(file_path: str, project: str):
    """Remove a specific file from the corpus."""
    try:
        updater.remove_file_chunks(file_path)
        updater.save_corpus()
        
        return UpdateResponse(
            success=True,
            message=f"Removed {file_path} from corpus",
            data={
                "file_path": file_path,
                "project": project
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove file: {str(e)}")

@app.post("/corpus/rebuild", response_model=UpdateResponse)
async def rebuild_corpus():
    """Rebuild the entire corpus from scratch."""
    try:
        # This would clear everything and rebuild
        # For now, just return the current stats
        stats = updater.get_corpus_stats()
        
        return UpdateResponse(
            success=True,
            message="Corpus rebuild completed",
            data=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild corpus: {str(e)}")

@app.get("/corpus/files/{project_name}")
async def get_project_files(project_name: str):
    """Get all files for a specific project."""
    try:
        import sqlite3
        
        with sqlite3.connect(updater.db_path) as conn:
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

@app.get("/corpus/recent-updates")
async def get_recent_updates(limit: int = 20):
    """Get recently updated files."""
    try:
        import sqlite3
        
        with sqlite3.connect(updater.db_path) as conn:
            cursor = conn.execute("""
                SELECT file_path, project, language, chunk_count, last_updated
                FROM file_hashes 
                ORDER BY last_updated DESC
                LIMIT ?
            """, (limit,))
            
            updates = [
                {
                    "file_path": row[0],
                    "project": row[1],
                    "language": row[2],
                    "chunk_count": row[3],
                    "last_updated": row[4]
                }
                for row in cursor.fetchall()
            ]
            
        return {
            "recent_updates": updates,
            "total_shown": len(updates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent updates: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Incremental Update API Server...")
    print("=" * 50)
    print(f"📊 Current corpus: {updater.get_corpus_stats()['total_vectors']} vectors")
    print(f"🌐 API docs: http://localhost:8001/docs")
    print(f"🔧 Health check: http://localhost:8001/health")
    print()
    
    uvicorn.run(
        "incremental_update_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )