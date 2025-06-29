#!/usr/bin/env python3
"""
Simple Corpus Expansion
Add high-quality repositories to existing corpus using OpenAI embeddings
Works with the current corpus format and structure
"""

import asyncio
import json
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import requests
from datetime import datetime

@dataclass
class CuratedRepo:
    """High-quality repository for corpus expansion."""
    name: str
    url: str
    primary_language: str
    estimated_vectors: int
    onedev_benefit: str
    quality_reason: str

class SimpleCorpusExpander:
    """Simple corpus expansion using existing data structures."""
    
    def __init__(self):
        self.corpus_file = "<project-root>/data/real_vector_corpus.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="corpus_exp_"))
        
        # Load existing corpus
        self.corpus_data = self.load_corpus()
        
        # High-quality, onedev-relevant repositories
        self.curated_repos = [
            CuratedRepo(
                "daisyui", 
                "https://github.com/saadeghi/daisyui",
                "javascript",
                300,
                "UI component patterns for onedev frontend",
                "Clean CSS components, excellent docs, 31k+ stars"
            ),
            CuratedRepo(
                "fastapi",
                "https://github.com/tiangolo/fastapi", 
                "python",
                400,
                "Modern API patterns for onedev backend",
                "Excellent type hints, async, 70k+ stars"
            ),
            CuratedRepo(
                "atproto",
                "https://github.com/bluesky-social/atproto",
                "typescript", 
                500,
                "Distributed protocol patterns",
                "Clean TypeScript, distributed systems"
            ),
            CuratedRepo(
                "trpc",
                "https://github.com/trpc/trpc",
                "typescript",
                300,
                "Type-safe API patterns for onedev",
                "End-to-end type safety, modern standard"
            ),
            CuratedRepo(
                "zod",
                "https://github.com/colinhacks/zod",
                "typescript",
                200,
                "Schema validation for onedev data",
                "Clean TypeScript, comprehensive validation"
            ),
        ]
        
    def load_corpus(self) -> Dict[str, Any]:
        """Load existing corpus data."""
        if not Path(self.corpus_file).exists():
            return {
                "metadata": {
                    "creation_date": datetime.now().isoformat(),
                    "source": "mixed",
                    "total_vectors": 0,
                    "vector_dimensions": 128,
                    "corpus_version": "2.0"
                },
                "vectors": []
            }
            
        with open(self.corpus_file, 'r') as f:
            data = json.load(f)
            
        print(f"📊 Loaded corpus: {data['metadata']['total_vectors']} vectors")
        return data
        
    def save_corpus(self):
        """Save updated corpus data."""
        # Update metadata
        self.corpus_data['metadata']['total_vectors'] = len(self.corpus_data['vectors'])
        self.corpus_data['metadata']['last_updated'] = datetime.now().isoformat()
        
        with open(self.corpus_file, 'w') as f:
            json.dump(self.corpus_data, f, indent=2)
            
        print(f"💾 Saved corpus: {self.corpus_data['metadata']['total_vectors']} vectors")
        
    def get_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from OpenAI API."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set - using mock embeddings")
            # Return mock 128-dimensional embedding
            np.random.seed(hash(text) % 2**32)
            return np.random.normal(0, 1, 128).tolist()
            
        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": text,
                    "model": "text-embedding-3-small",
                    "dimensions": 128
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['data'][0]['embedding']
            else:
                print(f"⚠️  OpenAI API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            return None
            
    def clone_repository(self, repo: CuratedRepo) -> Optional[Path]:
        """Clone repository with timeout."""
        print(f"📥 Cloning {repo.name}...")
        
        repo_dir = self.temp_dir / repo.name
        
        try:
            subprocess.run([
                'git', 'clone', '--depth', '1', '--single-branch',
                repo.url, str(repo_dir)
            ], check=True, capture_output=True, text=True, timeout=180)
            
            return repo_dir
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"❌ Clone failed: {repo.name}")
            return None
            
    def extract_code_chunks(self, file_path: Path, repo_name: str, language: str) -> List[Dict[str, Any]]:
        """Extract meaningful code chunks from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return []
            
        if len(content.strip()) < 100:
            return []
            
        # Simple chunking by functions/classes
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_start = 0
        
        # Patterns that indicate chunk boundaries
        chunk_patterns = {
            'javascript': ['function ', 'const ', 'class ', 'export '],
            'typescript': ['function ', 'const ', 'interface ', 'type ', 'class ', 'export '],
            'python': ['def ', 'class ', 'async def '],
            'css': ['.', '@'],
        }
        
        patterns = chunk_patterns.get(language, ['function ', 'const ', 'class '])
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if this line starts a new chunk
            is_chunk_start = any(stripped.startswith(pattern) for pattern in patterns)
            
            if is_chunk_start and current_chunk and len(current_chunk) > 5:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text.strip()) > 50:
                    rel_path = str(file_path.relative_to(file_path.parents[1]))  # Relative to repo
                    chunks.append({
                        'text': chunk_text,
                        'file_path': rel_path,
                        'start_line': current_start + 1,
                        'end_line': i,
                        'language': language,
                        'repo': repo_name
                    })
                
                # Start new chunk
                current_chunk = [line]
                current_start = i
            else:
                current_chunk.append(line)
                
        # Handle final chunk
        if current_chunk and len(current_chunk) > 5:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text.strip()) > 50:
                rel_path = str(file_path.relative_to(file_path.parents[1]))
                chunks.append({
                    'text': chunk_text,
                    'file_path': rel_path,
                    'start_line': current_start + 1,
                    'end_line': len(lines),
                    'language': language,
                    'repo': repo_name
                })
                
        return chunks[:20]  # Limit chunks per file
        
    def filter_quality_files(self, repo_dir: Path, language: str) -> List[Path]:
        """Find high-quality source files."""
        extensions = {
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'python': ['.py'],
            'css': ['.css', '.scss']
        }.get(language, ['.js', '.ts'])
        
        exclude_dirs = {
            'node_modules', '.git', 'dist', 'build', 'test', 'tests',
            'spec', '__tests__', 'docs', 'examples', 'demo'
        }
        
        files = []
        for ext in extensions:
            for file_path in repo_dir.rglob(f"*{ext}"):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                    
                # Skip test files
                if any(pattern in file_path.name.lower() for pattern in ['test', 'spec', 'mock']):
                    continue
                    
                # Check file size
                try:
                    size = file_path.stat().st_size
                    if 1000 < size < 100_000:  # 1KB to 100KB
                        files.append(file_path)
                except:
                    continue
                    
        return files[:50]  # Limit files per repo
        
    async def process_repository(self, repo: CuratedRepo) -> Dict[str, Any]:
        """Process a repository and add to corpus."""
        print(f"\n🔄 Processing: {repo.name}")
        print(f"   Benefit: {repo.onedev_benefit}")
        
        # Clone repo
        repo_dir = self.clone_repository(repo)
        if not repo_dir:
            return {'success': False, 'repo_name': repo.name}
            
        try:
            # Find quality files
            files = self.filter_quality_files(repo_dir, repo.primary_language)
            print(f"   Found {len(files)} quality files")
            
            if not files:
                return {'success': False, 'repo_name': repo.name}
                
            # Process files and extract chunks
            total_chunks = 0
            
            for file_path in files:
                chunks = self.extract_code_chunks(file_path, repo.name, repo.primary_language)
                
                for chunk in chunks:
                    # Generate embedding
                    embedding = self.get_openai_embedding(chunk['text'])
                    if not embedding:
                        continue
                        
                    # Create vector entry
                    vector_id = f"{repo.name}_{total_chunks}"
                    vector_entry = {
                        'id': vector_id,
                        'text': chunk['text'],
                        'file_path': chunk['file_path'],
                        'context_type': 'function',
                        'start_line': chunk['start_line'],
                        'end_line': chunk['end_line'],
                        'language': chunk['language'],
                        'project': repo.name,
                        'embedding': embedding
                    }
                    
                    self.corpus_data['vectors'].append(vector_entry)
                    total_chunks += 1
                    
                    if total_chunks % 10 == 0:
                        print(f"     {total_chunks} chunks processed")
                        
                    # Limit per repo
                    if total_chunks >= 200:
                        break
                        
                if total_chunks >= 200:
                    break
                    
            return {
                'success': True,
                'repo_name': repo.name,
                'chunks_added': total_chunks
            }
            
        finally:
            # Cleanup
            try:
                shutil.rmtree(repo_dir)
            except:
                pass
                
    async def expand_corpus(self) -> Dict[str, Any]:
        """Execute corpus expansion."""
        print("🎯 Simple Corpus Expansion")
        print("=" * 40)
        
        initial_count = len(self.corpus_data['vectors'])
        print(f"📊 Starting vectors: {initial_count:,}")
        
        print(f"\n📚 Repositories to add:")
        for i, repo in enumerate(self.curated_repos, 1):
            print(f"   {i}. {repo.name} - {repo.onedev_benefit}")
            
        # Process repositories
        results = []
        total_added = 0
        
        for repo in self.curated_repos:
            result = await self.process_repository(repo)
            results.append(result)
            
            if result['success']:
                added = result['chunks_added']
                total_added += added
                print(f"✅ {repo.name}: {added} chunks")
            else:
                print(f"❌ {repo.name}: failed")
                
            # Save progress
            self.save_corpus()
            current_count = len(self.corpus_data['vectors'])
            print(f"📊 Total vectors: {current_count:,}")
            
        final_count = len(self.corpus_data['vectors'])
        
        return {
            'initial_vectors': initial_count,
            'final_vectors': final_count,
            'vectors_added': total_added,
            'repos_processed': len([r for r in results if r['success']])
        }
        
    def cleanup(self):
        """Cleanup temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

async def main():
    """Main function."""
    expander = SimpleCorpusExpander()
    
    try:
        result = await expander.expand_corpus()
        
        print(f"\n🎉 Expansion Complete!")
        print(f"   Initial: {result['initial_vectors']:,} vectors")
        print(f"   Final: {result['final_vectors']:,} vectors") 
        print(f"   Added: {result['vectors_added']:,} vectors")
        print(f"   Repos: {result['repos_processed']}/5 successful")
        
    finally:
        expander.cleanup()

if __name__ == "__main__":
    asyncio.run(main())