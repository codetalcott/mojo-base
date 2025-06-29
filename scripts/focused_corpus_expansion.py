#!/usr/bin/env python3
"""
Focused Corpus Expansion
High-quality, actively maintained repositories that benefit onedev development
Curated selection based on quality standards and active maintenance
"""

import asyncio
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.corpus.incremental_updater import IncrementalUpdater

@dataclass
class CuratedRepo:
    """High-quality repository for corpus expansion."""
    name: str
    url: str
    primary_language: str
    estimated_vectors: int
    onedev_benefit: str
    quality_reason: str
    maintenance_status: str

class FocusedCorpusExpander:
    """Manages focused expansion with high-quality, relevant repositories."""
    
    def __init__(self):
        self.updater = IncrementalUpdater()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="focused_corpus_"))
        
        # Curated repositories that benefit onedev and meet quality standards
        self.curated_repos = [
            # DaisyUI - UI Component Library
            CuratedRepo(
                "daisyui", 
                "https://github.com/saadeghi/daisyui",
                "javascript",
                400,
                "UI component patterns for onedev's frontend development",
                "Clean CSS component architecture, excellent documentation",
                "Very active - 2024 commits, 31k+ stars"
            ),
            
            # FastAPI - Modern Python API Framework
            CuratedRepo(
                "fastapi",
                "https://github.com/tiangolo/fastapi", 
                "python",
                600,
                "Modern API patterns for onedev's backend services",
                "Excellent type hints, async support, auto-documentation",
                "Very active - maintained by Tiangolo, 70k+ stars"
            ),
            
            # AT Protocol - Distributed Social Protocol
            CuratedRepo(
                "atproto",
                "https://github.com/bluesky-social/atproto",
                "typescript", 
                800,
                "Distributed protocol patterns for onedev's architecture",
                "Clean TypeScript, well-designed distributed systems",
                "Very active - Bluesky team, production use"
            ),
            
            # Prisma - Modern Database Toolkit
            CuratedRepo(
                "prisma",
                "https://github.com/prisma/prisma",
                "typescript",
                1000,
                "Database patterns and type-safe queries for onedev",
                "Excellent TypeScript patterns, type safety focus",
                "Very active - commercial backing, 37k+ stars"
            ),
            
            # tRPC - Type-safe APIs
            CuratedRepo(
                "trpc",
                "https://github.com/trpc/trpc",
                "typescript",
                500,
                "Type-safe API patterns for onedev's client-server communication",
                "Excellent TypeScript, end-to-end type safety",
                "Very active - modern standard for type-safe APIs"
            ),
            
            # Zod - Schema Validation
            CuratedRepo(
                "zod",
                "https://github.com/colinhacks/zod",
                "typescript",
                300,
                "Schema validation patterns for onedev's data handling",
                "Clean TypeScript, comprehensive validation library",
                "Very active - widely adopted, excellent DX"
            ),
            
            # Drizzle ORM - Modern TypeScript ORM
            CuratedRepo(
                "drizzle-orm",
                "https://github.com/drizzle-team/drizzle-orm",
                "typescript",
                600,
                "Modern ORM patterns for onedev's database layer",
                "Excellent TypeScript, SQL-like syntax, type safety",
                "Very active - modern alternative to Prisma"
            ),
            
            # Vite - Modern Build Tool
            CuratedRepo(
                "vite",
                "https://github.com/vitejs/vite",
                "typescript",
                400,
                "Modern build tool patterns for onedev's frontend",
                "Fast development, excellent plugin architecture",
                "Very active - Evan You team, 64k+ stars"
            ),
            
            # Vitest - Fast Testing Framework
            CuratedRepo(
                "vitest",
                "https://github.com/vitest-dev/vitest",
                "typescript",
                300,
                "Modern testing patterns for onedev's test suite",
                "Fast, Vite-native, excellent TypeScript support",
                "Very active - part of Vite ecosystem"
            ),
            
            # TanStack Query - Data Fetching
            CuratedRepo(
                "tanstack-query",
                "https://github.com/TanStack/query",
                "typescript",
                500,
                "Advanced data fetching patterns for onedev's frontend",
                "Excellent caching, sync, background updates",
                "Very active - Tanner Linsley, widely adopted"
            ),
            
            # Tailwind CSS - Utility-first CSS
            CuratedRepo(
                "tailwindcss",
                "https://github.com/tailwindlabs/tailwindcss",
                "javascript",
                600,
                "Utility CSS patterns for onedev's styling system",
                "Clean architecture, excellent developer experience",
                "Very active - commercial backing, 78k+ stars"
            ),
            
            # Solid.js - Reactive UI Library
            CuratedRepo(
                "solid",
                "https://github.com/solidjs/solid",
                "typescript",
                400,
                "Reactive patterns for onedev's UI components",
                "Excellent performance, clean reactive model",
                "Very active - Ryan Carniato, growing adoption"
            ),
        ]
        
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current corpus statistics."""
        return self.updater.get_corpus_stats()
        
    def show_expansion_plan(self) -> Dict[str, Any]:
        """Show the focused expansion plan."""
        current_stats = self.get_current_stats()
        
        total_estimated = sum(repo.estimated_vectors for repo in self.curated_repos)
        
        return {
            'current_vectors': current_stats['total_vectors'],
            'current_projects': current_stats['projects'],
            'curated_repos': self.curated_repos,
            'estimated_addition': total_estimated,
            'estimated_final_total': current_stats['total_vectors'] + total_estimated
        }
        
    def clone_repository(self, repo: CuratedRepo) -> Optional[Path]:
        """Clone a repository to temporary directory."""
        print(f"📥 Cloning {repo.name}...")
        
        repo_dir = self.temp_dir / repo.name
        
        try:
            # Clone with depth 1 for speed, skip LFS
            subprocess.run([
                'git', 'clone', '--depth', '1', '--single-branch',
                '--no-tags', repo.url, str(repo_dir)
            ], check=True, capture_output=True, text=True, timeout=300)
            
            print(f"✅ Cloned {repo.name}")
            return repo_dir
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            error_msg = getattr(e, 'stderr', str(e))
            print(f"❌ Failed to clone {repo.name}: {error_msg}")
            return None
            
    def filter_high_quality_files(self, repo_dir: Path, language: str, repo_name: str) -> List[Path]:
        """Find high-quality source files, excluding tests and examples."""
        language_extensions = {
            'javascript': ['.js', '.jsx', '.mjs'],
            'typescript': ['.ts', '.tsx', '.mts'],
            'python': ['.py'],
            'css': ['.css', '.scss', '.sass'],
        }
        
        extensions = language_extensions.get(language, ['.js', '.ts'])
        
        code_files = []
        
        # Directories to exclude (tests, examples, docs, etc.)
        exclude_dirs = {
            'node_modules', '.git', 'dist', 'build', '__pycache__',
            'target', 'vendor', '.next', 'coverage', 'test', 'tests',
            'spec', 'specs', '__tests__', 'docs', 'documentation', 
            'examples', 'example', 'demo', 'demos', 'playground',
            'storybook', 'stories', '.storybook', 'e2e', 'cypress',
            'jest', 'vitest', 'benchmarks', 'benchmark', 'scripts',
            'tools', 'config', 'configs', '.github', '.vscode',
            'public', 'static', 'assets', 'images', 'img'
        }
        
        # Files to exclude
        exclude_files = {
            'test.', 'spec.', '.test.', '.spec.', 'mock.', '.mock.',
            'fixture.', '.fixture.', 'config.', '.config.', 'setup.',
            'rollup.', 'webpack.', 'vite.', 'jest.', 'babel.',
            'eslint.', 'prettier.', 'tsconfig.', 'tailwind.'
        }
        
        for ext in extensions:
            for file_path in repo_dir.rglob(f"*{ext}"):
                # Skip if in excluded directory
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                    
                # Skip excluded file patterns
                if any(pattern in file_path.name.lower() for pattern in exclude_files):
                    continue
                    
                # Skip if file is too large (>200KB) or too small (<100 bytes)
                try:
                    size = file_path.stat().st_size
                    if size > 200 * 1024 or size < 100:
                        continue
                except:
                    continue
                    
                # Prioritize source directories
                path_parts = file_path.parts
                priority = 0
                
                # Higher priority for core source directories
                if any(part in path_parts for part in ['src', 'lib', 'core', 'packages']):
                    priority += 100
                    
                # Medium priority for component/util directories  
                if any(part in path_parts for part in ['components', 'utils', 'helpers', 'hooks']):
                    priority += 50
                    
                code_files.append((priority, file_path))
                
        # Sort by priority and limit
        code_files.sort(key=lambda x: x[0], reverse=True)
        selected_files = [f[1] for f in code_files[:300]]  # Max 300 high-quality files
        
        print(f"   Selected {len(selected_files)} high-quality files from {repo_name}")
        return selected_files
        
    async def process_repository(self, repo: CuratedRepo) -> Dict[str, Any]:
        """Process a single curated repository."""
        print(f"\n🔄 Processing: {repo.name}")
        print(f"   Language: {repo.primary_language}")
        print(f"   OneDevBenefit: {repo.onedev_benefit}")
        print(f"   Quality: {repo.quality_reason}")
        print(f"   Maintenance: {repo.maintenance_status}")
        
        # Clone repository
        repo_dir = self.clone_repository(repo)
        if not repo_dir:
            return {'success': False, 'error': 'Clone failed', 'repo_name': repo.name}
            
        try:
            # Find high-quality code files
            code_files = self.filter_high_quality_files(repo_dir, repo.primary_language, repo.name)
            
            if not code_files:
                return {'success': False, 'error': 'No suitable files found', 'repo_name': repo.name}
                
            # Process files in batches
            batch_size = 5  # Smaller batches for quality focus
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
                            
                        # Skip files with minimal content
                        if len(content.strip()) < 100:
                            continue
                            
                        # Skip files that are mostly comments or whitespace
                        lines = content.split('\n')
                        code_lines = [l for l in lines if l.strip() and not l.strip().startswith(('*', '//', '#', '<!--'))]
                        if len(code_lines) < 10:
                            continue
                            
                        # Get relative path within repo
                        rel_path = str(file_path.relative_to(repo_dir))
                        
                        # Add to corpus
                        chunks = await self.updater.add_file_to_corpus(
                            rel_path, content, repo.name, repo.primary_language
                        )
                        
                        total_chunks += len(chunks)
                        processed_files += 1
                        
                        if processed_files % 5 == 0:
                            print(f"     {processed_files}/{len(code_files)} files → {total_chunks} chunks")
                            
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
                'onedev_benefit': repo.onedev_benefit,
                'errors': errors[:3]  # Show first 3 errors
            }
            
        finally:
            # Cleanup cloned repo
            try:
                shutil.rmtree(repo_dir)
            except:
                pass
                
    async def expand_corpus(self, max_repos: Optional[int] = None) -> Dict[str, Any]:
        """Execute focused corpus expansion."""
        print("🎯 Focused Corpus Expansion")
        print("High-quality repositories for onedev development")
        print("=" * 60)
        
        # Show expansion plan
        plan = self.show_expansion_plan()
        
        print(f"📊 Current State:")
        print(f"   Vectors: {plan['current_vectors']:,}")
        print(f"   Projects: {', '.join(plan['current_projects'])}")
        
        repos_to_process = self.curated_repos[:max_repos or len(self.curated_repos)]
        
        print(f"\n📚 Curated Repositories ({len(repos_to_process)} selected):")
        for i, repo in enumerate(repos_to_process, 1):
            print(f"   {i:2d}. {repo.name} ({repo.estimated_vectors:,} vectors)")
            print(f"       OneDevBenefit: {repo.onedev_benefit}")
            print(f"       Quality: {repo.quality_reason}")
            print(f"       Status: {repo.maintenance_status}")
            print()
            
        print(f"📈 Estimated Impact:")
        print(f"   Addition: ~{sum(r.estimated_vectors for r in repos_to_process):,} vectors")
        print(f"   Final total: ~{plan['estimated_final_total']:,} vectors")
        
        # Confirm before proceeding
        response = input(f"\nProceed with focused expansion? (y/n): ")
        if response.lower() != 'y':
            print("Expansion cancelled.")
            return plan
            
        # Process repositories
        results = []
        total_added = 0
        
        for i, repo in enumerate(repos_to_process, 1):
            print(f"\n[{i}/{len(repos_to_process)}] Processing {repo.name}...")
            
            result = await self.process_repository(repo)
            results.append(result)
            
            if result['success']:
                total_added += result['total_chunks']
                print(f"✅ Added {result['total_chunks']} chunks from {repo.name}")
                print(f"   OneDevBenefit: {result['onedev_benefit']}")
            else:
                print(f"❌ Failed: {repo.name} - {result.get('error', 'Unknown error')}")
                
            # Show progress
            current_stats = self.get_current_stats()
            print(f"📊 Progress: {current_stats['total_vectors']:,} total vectors")
                
        # Final results
        final_stats = self.get_current_stats()
        
        expansion_result = {
            'initial_vectors': plan['current_vectors'],
            'final_vectors': final_stats['total_vectors'],
            'vectors_added': total_added,
            'repos_processed': len([r for r in results if r['success']]),
            'repos_failed': len([r for r in results if not r['success']]),
            'processing_results': results,
            'final_stats': final_stats
        }
        
        print(f"\n🎉 Focused Corpus Expansion Complete!")
        print(f"   Initial: {expansion_result['initial_vectors']:,} vectors")
        print(f"   Final: {expansion_result['final_vectors']:,} vectors")
        print(f"   Added: {expansion_result['vectors_added']:,} vectors")
        print(f"   Success: {expansion_result['repos_processed']}/{len(repos_to_process)} repos")
        print(f"\n🎯 Quality Focus: High-maintenance repos benefiting onedev development")
        
        return expansion_result
        
    def cleanup(self):
        """Clean up temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

async def main():
    """Main focused expansion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Focused corpus expansion with curated high-quality repos")
    parser.add_argument('--max-repos', type=int, help='Maximum number of repositories to process')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    
    args = parser.parse_args()
    
    expander = FocusedCorpusExpander()
    
    try:
        if args.dry_run:
            plan = expander.show_expansion_plan()
            print("🔍 Focused Expansion Plan (Dry Run)")
            print("=" * 40)
            print(f"Current vectors: {plan['current_vectors']:,}")
            print(f"Current projects: {', '.join(plan['current_projects'])}")
            print(f"\nCurated repositories:")
            for i, repo in enumerate(plan['curated_repos'], 1):
                print(f"  {i:2d}. {repo.name} ({repo.estimated_vectors:,} vectors)")
                print(f"      OneDevBenefit: {repo.onedev_benefit}")
                print(f"      Quality: {repo.quality_reason}")
                print()
        else:
            await expander.expand_corpus(max_repos=args.max_repos)
    finally:
        expander.cleanup()

if __name__ == "__main__":
    asyncio.run(main())