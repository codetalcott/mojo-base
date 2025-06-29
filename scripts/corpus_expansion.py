#!/usr/bin/env python3
"""
Corpus Expansion Script
Systematically expand the semantic search corpus to 10k+ vectors
by mining popular GitHub repositories and open source projects
"""

import asyncio
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.corpus.incremental_updater import IncrementalUpdater

@dataclass
class RepoTarget:
    """Target repository for corpus expansion."""
    name: str
    url: str
    primary_language: str
    estimated_vectors: int
    priority: str  # 'high', 'medium', 'low'
    reason: str

class CorpusExpander:
    """Manages systematic expansion of the corpus to 10k+ vectors."""
    
    def __init__(self):
        self.updater = IncrementalUpdater()
        self.target_vectors = 10000
        self.temp_dir = Path(tempfile.mkdtemp(prefix="corpus_expansion_"))
        
        # High-quality repositories for corpus expansion
        self.repo_targets = [
            # React Ecosystem (High Priority)
            RepoTarget("react", "https://github.com/facebook/react", "javascript", 800, "high", 
                      "Core React library patterns"),
            RepoTarget("next.js", "https://github.com/vercel/next.js", "javascript", 1200, "high",
                      "Full-stack React framework patterns"),
            RepoTarget("material-ui", "https://github.com/mui/material-ui", "typescript", 600, "high",
                      "Component library patterns"),
            
            # Node.js/Backend (High Priority)
            RepoTarget("express", "https://github.com/expressjs/express", "javascript", 300, "high",
                      "Web server patterns"),
            RepoTarget("fastify", "https://github.com/fastify/fastify", "javascript", 400, "high",
                      "High-performance server patterns"),
            RepoTarget("nestjs", "https://github.com/nestjs/nest", "typescript", 800, "high",
                      "Enterprise Node.js patterns"),
            
            # Python Ecosystem (High Priority)
            RepoTarget("django", "https://github.com/django/django", "python", 1500, "high",
                      "Web framework patterns"),
            RepoTarget("flask", "https://github.com/pallets/flask", "python", 400, "high",
                      "Microframework patterns"),
            RepoTarget("fastapi", "https://github.com/tiangolo/fastapi", "python", 600, "high",
                      "Modern API patterns"),
            RepoTarget("pandas", "https://github.com/pandas-dev/pandas", "python", 2000, "medium",
                      "Data analysis patterns"),
            
            # JavaScript/TypeScript Utilities (Medium Priority)
            RepoTarget("lodash", "https://github.com/lodash/lodash", "javascript", 300, "medium",
                      "Utility function patterns"),
            RepoTarget("axios", "https://github.com/axios/axios", "javascript", 200, "medium",
                      "HTTP client patterns"),
            RepoTarget("moment", "https://github.com/moment/moment", "javascript", 150, "medium",
                      "Date/time handling patterns"),
            
            # Go Ecosystem (Medium Priority)
            RepoTarget("gin", "https://github.com/gin-gonic/gin", "go", 300, "medium",
                      "Go web framework patterns"),
            RepoTarget("echo", "https://github.com/labstack/echo", "go", 250, "medium",
                      "Go HTTP framework patterns"),
            RepoTarget("gorilla-mux", "https://github.com/gorilla/mux", "go", 150, "medium",
                      "Go routing patterns"),
            
            # Rust Ecosystem (Medium Priority)
            RepoTarget("actix-web", "https://github.com/actix/actix-web", "rust", 400, "medium",
                      "Rust web framework patterns"),
            RepoTarget("tokio", "https://github.com/tokio-rs/tokio", "rust", 600, "medium",
                      "Async runtime patterns"),
            RepoTarget("serde", "https://github.com/serde-rs/serde", "rust", 300, "medium",
                      "Serialization patterns"),
            
            # Developer Tools (Low Priority)
            RepoTarget("webpack", "https://github.com/webpack/webpack", "javascript", 800, "low",
                      "Build tool patterns"),
            RepoTarget("vite", "https://github.com/vitejs/vite", "typescript", 400, "low",
                      "Modern build tool patterns"),
            RepoTarget("prettier", "https://github.com/prettier/prettier", "javascript", 300, "low",
                      "Code formatting patterns"),
            
            # Example Apps/Templates (Low Priority)
            RepoTarget("create-react-app", "https://github.com/facebook/create-react-app", "javascript", 200, "low",
                      "React starter patterns"),
            RepoTarget("electron", "https://github.com/electron/electron", "javascript", 1000, "low",
                      "Desktop app patterns"),
        ]
        
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current corpus statistics."""
        return self.updater.get_corpus_stats()
        
    def calculate_expansion_plan(self) -> Dict[str, Any]:
        """Calculate how many vectors we need and which repos to target."""
        current_stats = self.get_current_stats()
        current_vectors = current_stats['total_vectors']
        vectors_needed = max(0, self.target_vectors - current_vectors)
        
        # Sort repos by priority and estimated vectors
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_repos = sorted(
            self.repo_targets,
            key=lambda r: (priority_order[r.priority], -r.estimated_vectors)
        )
        
        # Select repos to reach target
        selected_repos = []
        estimated_total = 0
        
        for repo in sorted_repos:
            if estimated_total >= vectors_needed:
                break
            selected_repos.append(repo)
            estimated_total += repo.estimated_vectors
            
        return {
            'current_vectors': current_vectors,
            'target_vectors': self.target_vectors,
            'vectors_needed': vectors_needed,
            'selected_repos': selected_repos,
            'estimated_total_addition': estimated_total,
            'estimated_final_total': current_vectors + estimated_total
        }
        
    def clone_repository(self, repo: RepoTarget) -> Optional[Path]:
        """Clone a repository to temporary directory."""
        print(f"📥 Cloning {repo.name}...")
        
        repo_dir = self.temp_dir / repo.name
        
        try:
            # Clone with depth 1 for speed
            subprocess.run([
                'git', 'clone', '--depth', '1', '--single-branch',
                repo.url, str(repo_dir)
            ], check=True, capture_output=True, text=True)
            
            print(f"✅ Cloned {repo.name} to {repo_dir}")
            return repo_dir
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone {repo.name}: {e.stderr}")
            return None
            
    def filter_code_files(self, repo_dir: Path, language: str) -> List[Path]:
        """Find relevant code files in repository."""
        language_extensions = {
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'python': ['.py'],
            'go': ['.go'],
            'rust': ['.rs'],
            'java': ['.java'],
            'cpp': ['.cpp', '.cc', '.cxx'],
            'c': ['.c', '.h']
        }
        
        extensions = language_extensions.get(language, ['.js', '.ts', '.py'])
        
        code_files = []
        
        # Common directories to exclude
        exclude_dirs = {
            'node_modules', '.git', 'dist', 'build', '__pycache__',
            'target', 'vendor', '.next', 'coverage', 'test',
            'tests', 'spec', 'docs', 'examples', 'demo'
        }
        
        for ext in extensions:
            for file_path in repo_dir.rglob(f"*{ext}"):
                # Skip if in excluded directory
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                    
                # Skip if file is too large (>100KB)
                try:
                    if file_path.stat().st_size > 100 * 1024:
                        continue
                except:
                    continue
                    
                code_files.append(file_path)
                
        # Limit to reasonable number of files
        return code_files[:500]  # Max 500 files per repo
        
    async def process_repository(self, repo: RepoTarget) -> Dict[str, Any]:
        """Process a single repository and add to corpus."""
        print(f"\n🔄 Processing repository: {repo.name}")
        print(f"   Language: {repo.primary_language}")
        print(f"   Expected vectors: {repo.estimated_vectors}")
        print(f"   Reason: {repo.reason}")
        
        # Clone repository
        repo_dir = self.clone_repository(repo)
        if not repo_dir:
            return {'success': False, 'error': 'Clone failed'}
            
        try:
            # Find code files
            code_files = self.filter_code_files(repo_dir, repo.primary_language)
            print(f"   Found {len(code_files)} code files")
            
            if not code_files:
                return {'success': False, 'error': 'No code files found'}
                
            # Process files in batches
            batch_size = 10
            total_chunks = 0
            processed_files = 0
            errors = []
            
            for i in range(0, len(code_files), batch_size):
                batch = code_files[i:i + batch_size]
                
                for file_path in batch:
                    try:
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Skip empty files
                        if len(content.strip()) < 50:
                            continue
                            
                        # Get relative path within repo
                        rel_path = str(file_path.relative_to(repo_dir))
                        
                        # Add to corpus
                        chunks = await self.updater.add_file_to_corpus(
                            rel_path, content, repo.name, repo.primary_language
                        )
                        
                        total_chunks += len(chunks)
                        processed_files += 1
                        
                        # Progress indicator
                        if processed_files % 10 == 0:
                            print(f"     Processed {processed_files}/{len(code_files)} files, {total_chunks} chunks")
                            
                    except Exception as e:
                        errors.append({'file': str(file_path), 'error': str(e)})
                        
                # Brief pause between batches
                await asyncio.sleep(0.1)
                
            # Save corpus after processing repo
            self.updater.save_corpus()
            
            return {
                'success': True,
                'repo_name': repo.name,
                'processed_files': processed_files,
                'total_chunks': total_chunks,
                'errors': errors[:5]  # Show first 5 errors
            }
            
        finally:
            # Cleanup cloned repo
            try:
                shutil.rmtree(repo_dir)
            except:
                pass
                
    async def expand_corpus(self, max_repos: Optional[int] = None) -> Dict[str, Any]:
        """Execute corpus expansion plan."""
        print("🚀 Starting Corpus Expansion")
        print("=" * 50)
        
        # Get expansion plan
        plan = self.calculate_expansion_plan()
        
        print(f"📊 Expansion Plan:")
        print(f"   Current vectors: {plan['current_vectors']:,}")
        print(f"   Target vectors: {plan['target_vectors']:,}")
        print(f"   Vectors needed: {plan['vectors_needed']:,}")
        print(f"   Selected repos: {len(plan['selected_repos'])}")
        print(f"   Estimated addition: {plan['estimated_total_addition']:,}")
        print(f"   Estimated final total: {plan['estimated_final_total']:,}")
        
        if plan['vectors_needed'] <= 0:
            print("\n✅ Target already reached!")
            return plan
            
        print(f"\n📋 Repositories to process:")
        for i, repo in enumerate(plan['selected_repos'][:max_repos or len(plan['selected_repos'])]):
            print(f"   {i+1:2d}. {repo.name} ({repo.estimated_vectors:,} vectors) - {repo.reason}")
            
        # Confirm before proceeding
        response = input(f"\nProceed with expansion? (y/n): ")
        if response.lower() != 'y':
            print("Expansion cancelled.")
            return plan
            
        # Process repositories
        results = []
        total_added = 0
        
        repos_to_process = plan['selected_repos'][:max_repos or len(plan['selected_repos'])]
        
        for i, repo in enumerate(repos_to_process, 1):
            print(f"\n[{i}/{len(repos_to_process)}] Processing {repo.name}...")
            
            result = await self.process_repository(repo)
            results.append(result)
            
            if result['success']:
                total_added += result['total_chunks']
                print(f"✅ Added {result['total_chunks']} chunks from {repo.name}")
            else:
                print(f"❌ Failed to process {repo.name}: {result.get('error', 'Unknown error')}")
                
            # Show progress
            current_stats = self.get_current_stats()
            print(f"📊 Progress: {current_stats['total_vectors']:,} vectors total")
            
            # Check if we've reached target
            if current_stats['total_vectors'] >= self.target_vectors:
                print(f"\n🎉 Target reached! {current_stats['total_vectors']:,} vectors")
                break
                
        # Final stats
        final_stats = self.get_current_stats()
        
        expansion_result = {
            'initial_vectors': plan['current_vectors'],
            'final_vectors': final_stats['total_vectors'],
            'vectors_added': total_added,
            'repos_processed': len([r for r in results if r['success']]),
            'repos_failed': len([r for r in results if not r['success']]),
            'target_reached': final_stats['total_vectors'] >= self.target_vectors,
            'processing_results': results,
            'final_stats': final_stats
        }
        
        print(f"\n🎯 Corpus Expansion Complete!")
        print(f"   Initial vectors: {expansion_result['initial_vectors']:,}")
        print(f"   Final vectors: {expansion_result['final_vectors']:,}")
        print(f"   Vectors added: {expansion_result['vectors_added']:,}")
        print(f"   Repos processed: {expansion_result['repos_processed']}")
        print(f"   Target reached: {'✅' if expansion_result['target_reached'] else '❌'}")
        
        return expansion_result
        
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

async def main():
    """Main expansion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Expand corpus to 10k+ vectors")
    parser.add_argument('--max-repos', type=int, help='Maximum number of repositories to process')
    parser.add_argument('--target', type=int, default=10000, help='Target number of vectors')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    
    args = parser.parse_args()
    
    expander = CorpusExpander()
    expander.target_vectors = args.target
    
    try:
        if args.dry_run:
            plan = expander.calculate_expansion_plan()
            print("🔍 Expansion Plan (Dry Run)")
            print("=" * 30)
            print(f"Current vectors: {plan['current_vectors']:,}")
            print(f"Target vectors: {plan['target_vectors']:,}")
            print(f"Vectors needed: {plan['vectors_needed']:,}")
            print(f"\nSelected repositories:")
            for i, repo in enumerate(plan['selected_repos'], 1):
                print(f"  {i:2d}. {repo.name} ({repo.estimated_vectors:,} vectors)")
                print(f"      {repo.reason}")
        else:
            await expander.expand_corpus(max_repos=args.max_repos)
    finally:
        expander.cleanup()

if __name__ == "__main__":
    asyncio.run(main())