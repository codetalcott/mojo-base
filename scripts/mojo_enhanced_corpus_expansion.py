#!/usr/bin/env python3
"""
Mojo-Enhanced Corpus Expansion
Includes Mojo language patterns alongside other high-quality repositories
Special handling for .mojo and .🔥 files for comprehensive language support
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
    special_handling: bool = False

class MojoEnhancedCorpusExpander:
    """Corpus expansion with special Mojo language support."""
    
    def __init__(self):
        self.corpus_file = "/Users/williamtalcott/projects/mojo-base/data/real_vector_corpus.json"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="mojo_corpus_"))
        
        # Load existing corpus
        self.corpus_data = self.load_corpus()
        
        # Enhanced repository list with Mojo included
        self.curated_repos = [
            # Mojo Language - High Priority
            CuratedRepo(
                "mojo", 
                "https://github.com/modularml/mojo",
                "mojo",
                600,
                "Mojo language patterns, syntax, and stdlib examples",
                "Official Mojo repo from Modular, cutting-edge language design",
                special_handling=True
            ),
            
            # DaisyUI - UI Framework
            CuratedRepo(
                "daisyui", 
                "https://github.com/saadeghi/daisyui",
                "javascript",
                300,
                "UI component patterns for onedev frontend",
                "Clean CSS components, excellent docs, 31k+ stars"
            ),
            
            # FastAPI - Python Framework
            CuratedRepo(
                "fastapi",
                "https://github.com/tiangolo/fastapi", 
                "python",
                400,
                "Modern API patterns for onedev backend",
                "Excellent type hints, async patterns, 70k+ stars"
            ),
            
            # AT Protocol - Distributed Systems
            CuratedRepo(
                "atproto",
                "https://github.com/bluesky-social/atproto",
                "typescript", 
                500,
                "Distributed protocol patterns for onedev architecture",
                "Clean TypeScript, well-designed distributed systems"
            ),
            
            # tRPC - Type-safe APIs
            CuratedRepo(
                "trpc",
                "https://github.com/trpc/trpc",
                "typescript",
                300,
                "Type-safe API patterns for onedev client-server communication",
                "End-to-end type safety, modern API standard"
            ),
            
            # Zod - Schema Validation
            CuratedRepo(
                "zod",
                "https://github.com/colinhacks/zod",
                "typescript",
                200,
                "Schema validation patterns for onedev data handling",
                "Clean TypeScript, comprehensive validation library"
            ),
            
            # Prisma - Database Toolkit
            CuratedRepo(
                "prisma",
                "https://github.com/prisma/prisma",
                "typescript",
                400,
                "Database patterns and type-safe queries for onedev",
                "Excellent TypeScript patterns, type safety focus"
            ),
            
            # Drizzle ORM - Modern TypeScript ORM
            CuratedRepo(
                "drizzle-orm",
                "https://github.com/drizzle-team/drizzle-orm",
                "typescript",
                300,
                "Modern ORM patterns for onedev's database layer",
                "Excellent TypeScript, SQL-like syntax, type safety"
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
        """Get embedding from OpenAI API or generate mock."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            # Generate deterministic mock embedding based on text hash
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
            # For Mojo repo, we need to be more careful with size
            if repo.name == "mojo":
                subprocess.run([
                    'git', 'clone', '--depth', '1', '--single-branch',
                    '--filter=blob:limit=1m',  # Limit large files
                    repo.url, str(repo_dir)
                ], check=True, capture_output=True, text=True, timeout=300)
            else:
                subprocess.run([
                    'git', 'clone', '--depth', '1', '--single-branch',
                    repo.url, str(repo_dir)
                ], check=True, capture_output=True, text=True, timeout=180)
            
            return repo_dir
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"❌ Clone failed: {repo.name}")
            return None
            
    def extract_mojo_chunks(self, file_path: Path, repo_name: str) -> List[Dict[str, Any]]:
        """Extract code chunks from Mojo files with special handling."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return []
            
        if len(content.strip()) < 50:
            return []
            
        # Mojo-specific patterns
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_start = 0
        
        # Mojo patterns for chunking
        mojo_patterns = [
            'fn ', 'def ', 'struct ', 'trait ', 'alias ',
            'var ', 'let ', 'import ', 'from '
        ]
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for Mojo-specific chunk boundaries
            is_chunk_start = any(stripped.startswith(pattern) for pattern in mojo_patterns)
            
            if is_chunk_start and current_chunk and len(current_chunk) > 3:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text.strip()) > 30:  # Smaller minimum for Mojo
                    rel_path = str(file_path.relative_to(file_path.parents[1]))
                    chunks.append({
                        'text': chunk_text,
                        'file_path': rel_path,
                        'start_line': current_start + 1,
                        'end_line': i,
                        'language': 'mojo',
                        'repo': repo_name
                    })
                
                # Start new chunk
                current_chunk = [line]
                current_start = i
            else:
                current_chunk.append(line)
                
        # Handle final chunk
        if current_chunk and len(current_chunk) > 3:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text.strip()) > 30:
                rel_path = str(file_path.relative_to(file_path.parents[1]))
                chunks.append({
                    'text': chunk_text,
                    'file_path': rel_path,
                    'start_line': current_start + 1,
                    'end_line': len(lines),
                    'language': 'mojo',
                    'repo': repo_name
                })
                
        return chunks[:30]  # More chunks for Mojo due to importance
        
    def extract_code_chunks(self, file_path: Path, repo_name: str, language: str) -> List[Dict[str, Any]]:
        """Extract meaningful code chunks from a file."""
        # Special handling for Mojo files
        if language == 'mojo':
            return self.extract_mojo_chunks(file_path, repo_name)
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return []
            
        if len(content.strip()) < 100:
            return []
            
        # Standard chunking for other languages
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_start = 0
        
        # Language-specific patterns
        chunk_patterns = {
            'javascript': ['function ', 'const ', 'class ', 'export ', 'import '],
            'typescript': ['function ', 'const ', 'interface ', 'type ', 'class ', 'export '],
            'python': ['def ', 'class ', 'async def ', 'import ', 'from '],
            'css': ['.', '@', ':root'],
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
                    rel_path = str(file_path.relative_to(file_path.parents[1]))
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
        
    def filter_quality_files(self, repo_dir: Path, language: str, repo_name: str) -> List[Path]:
        """Find high-quality source files with Mojo support."""
        extensions = {
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'python': ['.py'],
            'css': ['.css', '.scss'],
            'mojo': ['.mojo', '.🔥']  # Mojo files
        }.get(language, ['.js', '.ts'])
        
        exclude_dirs = {
            'node_modules', '.git', 'dist', 'build', 'test', 'tests',
            'spec', '__tests__', 'docs', 'examples', 'demo', 'benchmark',
            'benchmarks', '.github', 'scripts', 'tools'
        }
        
        # For Mojo repo, be more selective
        if repo_name == "mojo":
            exclude_dirs.update({
                'third-party', 'external', 'vendor', 'cmake', 'llvm',
                'utils', 'packaging', 'docker'
            })
        
        files = []
        for ext in extensions:
            for file_path in repo_dir.rglob(f"*{ext}"):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                    
                # Skip test files
                if any(pattern in file_path.name.lower() for pattern in ['test', 'spec', 'mock', 'benchmark']):
                    continue
                    
                # Check file size
                try:
                    size = file_path.stat().st_size
                    min_size = 500 if language == 'mojo' else 1000  # Smaller min for Mojo
                    max_size = 50_000 if language == 'mojo' else 100_000  # Smaller max for Mojo
                    
                    if min_size < size < max_size:
                        files.append(file_path)
                except:
                    continue
                    
        # For Mojo, prioritize stdlib and core files
        if repo_name == "mojo":
            priority_files = []
            other_files = []
            
            for file_path in files:
                path_str = str(file_path)
                if any(priority in path_str for priority in ['stdlib', 'core', 'builtin', 'collections']):
                    priority_files.append(file_path)
                else:
                    other_files.append(file_path)
                    
            # Return priority files first, then others
            files = priority_files + other_files
                    
        return files[:80 if repo_name == "mojo" else 50]  # More files for Mojo
        
    async def process_repository(self, repo: CuratedRepo) -> Dict[str, Any]:
        """Process a repository and add to corpus."""
        print(f"\n🔄 Processing: {repo.name}")
        print(f"   Language: {repo.primary_language}")
        print(f"   Benefit: {repo.onedev_benefit}")
        
        if repo.special_handling:
            print(f"   🎯 Special handling enabled for {repo.name}")
        
        # Clone repo
        repo_dir = self.clone_repository(repo)
        if not repo_dir:
            return {'success': False, 'repo_name': repo.name}
            
        try:
            # Find quality files
            files = self.filter_quality_files(repo_dir, repo.primary_language, repo.name)
            print(f"   Found {len(files)} quality files")
            
            if not files:
                return {'success': False, 'repo_name': repo.name}
                
            # Process files and extract chunks
            total_chunks = 0
            max_chunks = 400 if repo.name == "mojo" else 200  # More chunks for Mojo
            
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
                    
                    if total_chunks % 25 == 0:
                        print(f"     {total_chunks} chunks processed")
                        
                    # Limit per repo
                    if total_chunks >= max_chunks:
                        break
                        
                if total_chunks >= max_chunks:
                    break
                    
            return {
                'success': True,
                'repo_name': repo.name,
                'chunks_added': total_chunks,
                'language': repo.primary_language
            }
            
        finally:
            # Cleanup
            try:
                shutil.rmtree(repo_dir)
            except:
                pass
                
    async def expand_corpus(self, include_mojo: bool = True) -> Dict[str, Any]:
        """Execute Mojo-enhanced corpus expansion."""
        print("🔥 Mojo-Enhanced Corpus Expansion")
        print("=" * 45)
        
        initial_count = len(self.corpus_data['vectors'])
        print(f"📊 Starting vectors: {initial_count:,}")
        
        # Filter repos based on include_mojo flag
        repos_to_process = self.curated_repos
        if not include_mojo:
            repos_to_process = [r for r in self.curated_repos if r.name != "mojo"]
        
        print(f"\n📚 Repositories to add:")
        for i, repo in enumerate(repos_to_process, 1):
            icon = "🔥" if repo.name == "mojo" else "📦"
            print(f"   {i}. {icon} {repo.name} ({repo.primary_language})")
            print(f"      {repo.onedev_benefit}")
            
        print(f"\n🎯 Special focus on Mojo language patterns for semantic search enhancement")
        
        # Process repositories
        results = []
        total_added = 0
        languages_added = set()
        
        for repo in repos_to_process:
            result = await self.process_repository(repo)
            results.append(result)
            
            if result['success']:
                added = result['chunks_added']
                total_added += added
                languages_added.add(result['language'])
                
                icon = "🔥" if repo.name == "mojo" else "✅"
                print(f"{icon} {repo.name}: {added} chunks")
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
            'repos_processed': len([r for r in results if r['success']]),
            'languages_added': list(languages_added),
            'mojo_included': include_mojo and any(r['repo_name'] == 'mojo' and r['success'] for r in results)
        }
        
    def cleanup(self):
        """Cleanup temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mojo-enhanced corpus expansion")
    parser.add_argument('--skip-mojo', action='store_true', help='Skip Mojo repository')
    parser.add_argument('--mojo-only', action='store_true', help='Only process Mojo repository')
    
    args = parser.parse_args()
    
    expander = MojoEnhancedCorpusExpander()
    
    # Filter repos based on arguments
    if args.mojo_only:
        expander.curated_repos = [r for r in expander.curated_repos if r.name == "mojo"]
    elif args.skip_mojo:
        expander.curated_repos = [r for r in expander.curated_repos if r.name != "mojo"]
    
    try:
        result = await expander.expand_corpus(include_mojo=not args.skip_mojo)
        
        print(f"\n🎉 Mojo-Enhanced Expansion Complete!")
        print(f"   Initial: {result['initial_vectors']:,} vectors")
        print(f"   Final: {result['final_vectors']:,} vectors") 
        print(f"   Added: {result['vectors_added']:,} vectors")
        print(f"   Repos: {result['repos_processed']}/{len(expander.curated_repos)} successful")
        print(f"   Languages: {', '.join(result['languages_added'])}")
        
        if result.get('mojo_included'):
            print(f"   🔥 Mojo patterns included for enhanced semantic search!")
        
    finally:
        expander.cleanup()

if __name__ == "__main__":
    asyncio.run(main())