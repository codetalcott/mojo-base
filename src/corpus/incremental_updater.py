#!/usr/bin/env python3
"""
Incremental Corpus Updater
Efficiently add new code snippets to the semantic search corpus
without rebuilding the entire vector database
"""

import os
import sys
import json
import time
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import sqlite3
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.corpus.corpus_builder import CorpusBuilder

@dataclass
class CodeUpdate:
    """Represents a code update operation."""
    operation: str  # 'add', 'update', 'delete'
    file_path: str
    project: str
    language: str
    content: str
    chunk_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    hash: Optional[str] = None

@dataclass
class CorpusMetadata:
    """Metadata about corpus state."""
    total_vectors: int
    projects: List[str]
    languages: Set[str]
    last_updated: datetime
    version: str

class IncrementalUpdater:
    """Manages incremental updates to the semantic search corpus."""
    
    def __init__(self, corpus_dir: str = "data/real_corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.corpus_builder = CorpusBuilder()
        
        # Database for tracking changes
        self.db_path = self.corpus_dir / "corpus_metadata.db"
        self.init_database()
        
        # Load current corpus state
        self.vectors_file = self.corpus_dir / "vectors.npy"
        self.metadata_file = self.corpus_dir / "metadata.json"
        self.load_corpus()
        
    def init_database(self):
        """Initialize SQLite database for tracking file changes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_hashes (
                    file_path TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    language TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corpus_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    project TEXT NOT NULL,
                    language TEXT NOT NULL,
                    start_line INTEGER,
                    end_line INTEGER,
                    content_hash TEXT NOT NULL,
                    vector_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_path) REFERENCES file_hashes (file_path)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path ON corpus_chunks (file_path);
                CREATE INDEX IF NOT EXISTS idx_project ON corpus_chunks (project);
                CREATE INDEX IF NOT EXISTS idx_language ON corpus_chunks (language);
            """)
            
    def load_corpus(self):
        """Load existing corpus data."""
        self.vectors = None
        self.metadata = []
        
        if self.vectors_file.exists() and self.metadata_file.exists():
            try:
                self.vectors = np.load(self.vectors_file)
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded corpus: {len(self.metadata)} vectors")
            except Exception as e:
                print(f"⚠️  Error loading corpus: {e}")
                self.vectors = None
                self.metadata = []
        else:
            print("📂 No existing corpus found - will create new one")
            
    def save_corpus(self):
        """Save corpus data to disk."""
        if self.vectors is not None:
            np.save(self.vectors_file, self.vectors)
            
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
            
        print(f"💾 Saved corpus: {len(self.metadata)} vectors")
        
    def calculate_file_hash(self, content: str) -> str:
        """Calculate hash for file content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
        
    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get stored hash for file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT content_hash FROM file_hashes WHERE file_path = ?",
                (file_path,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
            
    def has_file_changed(self, file_path: str, content: str) -> bool:
        """Check if file content has changed."""
        current_hash = self.calculate_file_hash(content)
        stored_hash = self.get_file_hash(file_path)
        return stored_hash != current_hash
        
    def get_existing_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """Get existing chunks for a file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT chunk_id, start_line, end_line, vector_index
                FROM corpus_chunks 
                WHERE file_path = ?
                ORDER BY start_line
            """, (file_path,))
            
            return [
                {
                    'chunk_id': row[0],
                    'start_line': row[1], 
                    'end_line': row[2],
                    'vector_index': row[3]
                }
                for row in cursor.fetchall()
            ]
            
    def remove_file_chunks(self, file_path: str):
        """Remove all chunks for a file from corpus."""
        existing_chunks = self.get_existing_chunks(file_path)
        
        if not existing_chunks:
            return
            
        # Get vector indices to remove
        vector_indices = [chunk['vector_index'] for chunk in existing_chunks]
        
        # Remove from vectors and metadata
        if self.vectors is not None and len(vector_indices) > 0:
            # Create mask to keep vectors NOT in the removal list
            keep_mask = np.ones(len(self.vectors), dtype=bool)
            keep_mask[vector_indices] = False
            
            # Filter vectors and metadata
            self.vectors = self.vectors[keep_mask]
            self.metadata = [
                meta for i, meta in enumerate(self.metadata) 
                if i not in vector_indices
            ]
            
            # Update vector indices in database for remaining chunks
            self._reindex_database_after_removal(vector_indices)
            
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM corpus_chunks WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM file_hashes WHERE file_path = ?", (file_path,))
            
        print(f"🗑️  Removed {len(existing_chunks)} chunks from {file_path}")
        
    def _reindex_database_after_removal(self, removed_indices: List[int]):
        """Update vector indices in database after removal."""
        with sqlite3.connect(self.db_path) as conn:
            # Get all chunks with their current vector indices
            cursor = conn.execute("SELECT chunk_id, vector_index FROM corpus_chunks ORDER BY vector_index")
            all_chunks = cursor.fetchall()
            
            # Calculate new indices
            for chunk_id, old_index in all_chunks:
                # Count how many removed indices are before this one
                removed_before = sum(1 for ri in removed_indices if ri < old_index)
                new_index = old_index - removed_before
                
                # Update database
                conn.execute(
                    "UPDATE corpus_chunks SET vector_index = ? WHERE chunk_id = ?",
                    (new_index, chunk_id)
                )
                
    async def add_file_to_corpus(self, file_path: str, content: str, project: str, language: str) -> List[Dict[str, Any]]:
        """Add or update a file in the corpus."""
        print(f"📄 Processing: {file_path}")
        
        # Check if file has changed
        if not self.has_file_changed(file_path, content):
            print(f"⏭️  Skipped: {file_path} (unchanged)")
            return []
            
        # Remove existing chunks for this file
        self.remove_file_chunks(file_path)
        
        # Generate new chunks
        chunks = self.corpus_builder.extract_code_chunks(content, file_path, project, language)
        
        if not chunks:
            print(f"⚠️  No chunks extracted from {file_path}")
            return []
            
        # Generate embeddings for new chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = await self.embedding_generator.generate_embeddings_batch(chunk_texts)
        
        # Add to corpus
        added_chunks = []
        start_index = len(self.metadata) if self.metadata else 0
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_index = start_index + i
            chunk_id = f"{project}_{Path(file_path).stem}_{chunk['start_line']}_{chunk['end_line']}"
            
            # Add to vectors and metadata
            if self.vectors is None:
                self.vectors = embedding.reshape(1, -1)
            else:
                self.vectors = np.vstack([self.vectors, embedding.reshape(1, -1)])
                
            self.metadata.append({
                'id': chunk_id,
                'text': chunk['text'],
                'file_path': file_path,
                'project': project,
                'language': language,
                'context_type': chunk['context_type'],
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line'],
                'added_at': datetime.now().isoformat()
            })
            
            added_chunks.append({
                'chunk_id': chunk_id,
                'vector_index': vector_index,
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line']
            })
            
        # Update database
        content_hash = self.calculate_file_hash(content)
        
        with sqlite3.connect(self.db_path) as conn:
            # Update file hash
            conn.execute("""
                INSERT OR REPLACE INTO file_hashes 
                (file_path, project, language, content_hash, chunk_count)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path, project, language, content_hash, len(added_chunks)))
            
            # Add chunk records
            for chunk in added_chunks:
                conn.execute("""
                    INSERT INTO corpus_chunks 
                    (chunk_id, file_path, project, language, start_line, end_line, 
                     content_hash, vector_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk['chunk_id'], file_path, project, language,
                    chunk['start_line'], chunk['end_line'], content_hash, 
                    chunk['vector_index']
                ))
                
        print(f"✅ Added {len(added_chunks)} chunks from {file_path}")
        return added_chunks
        
    async def scan_project_for_updates(self, project_path: str, project_name: str) -> Dict[str, Any]:
        """Scan a project directory for file changes and update corpus."""
        project_path = Path(project_path)
        
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
            
        print(f"🔍 Scanning project: {project_name}")
        
        # Supported file extensions
        code_extensions = {'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.cpp', '.c', '.h'}
        
        # Find all code files
        code_files = []
        for ext in code_extensions:
            code_files.extend(project_path.rglob(f"*{ext}"))
            
        print(f"📁 Found {len(code_files)} code files")
        
        # Process files
        updates = {
            'added': [],
            'updated': [],
            'errors': []
        }
        
        for file_path in code_files:
            try:
                # Skip files that are too large
                if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                    continue
                    
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Determine language
                language = self.corpus_builder.detect_language(file_path.suffix)
                
                # Add to corpus
                relative_path = str(file_path.relative_to(project_path))
                chunks = await self.add_file_to_corpus(
                    relative_path, content, project_name, language
                )
                
                if chunks:
                    updates['added'].append({
                        'file_path': relative_path,
                        'chunks_added': len(chunks)
                    })
                    
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                updates['errors'].append({
                    'file_path': str(file_path),
                    'error': str(e)
                })
                
        # Save updated corpus
        self.save_corpus()
        
        return updates
        
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get current corpus statistics."""
        if not self.metadata:
            return {'total_vectors': 0, 'projects': [], 'languages': []}
            
        projects = list(set(item['project'] for item in self.metadata))
        languages = list(set(item['language'] for item in self.metadata))
        
        # Get database stats
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM file_hashes")
            file_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM corpus_chunks")
            chunk_count = cursor.fetchone()[0]
            
        return {
            'total_vectors': len(self.metadata),
            'total_files': file_count,
            'total_chunks': chunk_count,
            'projects': projects,
            'languages': languages,
            'corpus_size_mb': (self.vectors.nbytes / 1024 / 1024) if self.vectors is not None else 0
        }
        
    async def bulk_update_from_projects(self, projects: List[Dict[str, str]]) -> Dict[str, Any]:
        """Update corpus from multiple projects."""
        print(f"🚀 Starting bulk update for {len(projects)} projects")
        
        all_updates = {
            'projects_processed': 0,
            'total_files_added': 0,
            'total_chunks_added': 0,
            'errors': []
        }
        
        for project in projects:
            try:
                project_name = project['name']
                project_path = project['path']
                
                print(f"\n📦 Processing project: {project_name}")
                updates = await self.scan_project_for_updates(project_path, project_name)
                
                all_updates['projects_processed'] += 1
                all_updates['total_files_added'] += len(updates['added'])
                all_updates['total_chunks_added'] += sum(
                    update['chunks_added'] for update in updates['added']
                )
                all_updates['errors'].extend(updates['errors'])
                
            except Exception as e:
                print(f"❌ Error processing project {project.get('name', 'unknown')}: {e}")
                all_updates['errors'].append({
                    'project': project.get('name', 'unknown'),
                    'error': str(e)
                })
                
        # Final stats
        final_stats = self.get_corpus_stats()
        all_updates['final_corpus_stats'] = final_stats
        
        print(f"\n🎉 Bulk update completed!")
        print(f"   Projects processed: {all_updates['projects_processed']}")
        print(f"   Files added: {all_updates['total_files_added']}")
        print(f"   Chunks added: {all_updates['total_chunks_added']}")
        print(f"   Final corpus size: {final_stats['total_vectors']} vectors")
        
        return all_updates

async def main():
    """Main function for testing incremental updates."""
    updater = IncrementalUpdater()
    
    # Get current stats
    stats = updater.get_corpus_stats()
    print(f"📊 Current corpus: {stats['total_vectors']} vectors from {len(stats['projects'])} projects")
    
    # Example: Add a new project
    print("\n🧪 Testing incremental update...")
    
    # You can add projects like this:
    # projects = [
    #     {'name': 'new-project', 'path': '/path/to/new/project'},
    # ]
    # updates = await updater.bulk_update_from_projects(projects)
    
    print("✅ Incremental updater ready for use!")

if __name__ == "__main__":
    asyncio.run(main())