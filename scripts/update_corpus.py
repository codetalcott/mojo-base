#!/usr/bin/env python3
"""
Corpus Update Script
Command-line tool for managing incremental corpus updates
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.corpus.incremental_updater import IncrementalUpdater

def print_banner():
    """Print script banner."""
    print("🚀 Mojo Semantic Search - Corpus Updater")
    print("=" * 50)

def print_stats(stats: Dict[str, Any]):
    """Print corpus statistics."""
    print(f"📊 Corpus Statistics:")
    print(f"   Total vectors: {stats['total_vectors']:,}")
    print(f"   Total files: {stats['total_files']:,}")
    print(f"   Total chunks: {stats['total_chunks']:,}")
    print(f"   Projects: {len(stats['projects'])} ({', '.join(stats['projects'])})")
    print(f"   Languages: {len(stats['languages'])} ({', '.join(stats['languages'])})")
    print(f"   Corpus size: {stats['corpus_size_mb']:.1f} MB")

async def add_single_file(updater: IncrementalUpdater, args):
    """Add a single file to corpus."""
    print(f"📄 Adding file: {args.file}")
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ File not found: {args.file}")
        return
        
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
        
    # Determine language
    language = args.language
    if not language:
        language = updater.corpus_builder.detect_language(file_path.suffix)
        
    # Add to corpus
    chunks = await updater.add_file_to_corpus(
        str(file_path), content, args.project, language
    )
    
    updater.save_corpus()
    
    print(f"✅ Added {len(chunks)} chunks from {args.file}")
    print_stats(updater.get_corpus_stats())

async def add_project(updater: IncrementalUpdater, args):
    """Add entire project to corpus."""
    print(f"📦 Adding project: {args.name} from {args.path}")
    
    project_path = Path(args.path)
    if not project_path.exists():
        print(f"❌ Project path not found: {args.path}")
        return
        
    # Scan and add project
    updates = await updater.scan_project_for_updates(args.path, args.name)
    
    print(f"✅ Project added:")
    print(f"   Files processed: {len(updates['added'])}")
    print(f"   Total chunks: {sum(u['chunks_added'] for u in updates['added'])}")
    
    if updates['errors']:
        print(f"⚠️  Errors encountered: {len(updates['errors'])}")
        for error in updates['errors'][:5]:  # Show first 5 errors
            print(f"     {error['file_path']}: {error['error']}")
            
    print_stats(updater.get_corpus_stats())

async def bulk_update(updater: IncrementalUpdater, args):
    """Perform bulk update from config file."""
    print(f"📋 Bulk update from: {args.config}")
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {args.config}")
        return
        
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return
        
    projects = config.get('projects', [])
    if not projects:
        print("❌ No projects found in config file")
        return
        
    print(f"🚀 Processing {len(projects)} projects...")
    
    # Perform bulk update
    updates = await updater.bulk_update_from_projects(projects)
    
    print(f"✅ Bulk update completed:")
    print(f"   Projects processed: {updates['projects_processed']}")
    print(f"   Files added: {updates['total_files_added']}")
    print(f"   Chunks added: {updates['total_chunks_added']}")
    
    if updates['errors']:
        print(f"⚠️  Errors: {len(updates['errors'])}")
        
    print_stats(updates['final_corpus_stats'])

async def remove_project(updater: IncrementalUpdater, args):
    """Remove project from corpus."""
    print(f"🗑️  Removing project: {args.name}")
    
    import sqlite3
    removed_count = 0
    
    with sqlite3.connect(updater.db_path) as conn:
        cursor = conn.execute(
            "SELECT DISTINCT file_path FROM file_hashes WHERE project = ?",
            (args.name,)
        )
        file_paths = [row[0] for row in cursor.fetchall()]
        
        if not file_paths:
            print(f"⚠️  No files found for project: {args.name}")
            return
            
        # Remove each file
        for file_path in file_paths:
            updater.remove_file_chunks(file_path)
            removed_count += 1
            
    updater.save_corpus()
    
    print(f"✅ Removed {removed_count} files from project {args.name}")
    print_stats(updater.get_corpus_stats())

async def list_projects(updater: IncrementalUpdater, args):
    """List all projects in corpus."""
    stats = updater.get_corpus_stats()
    
    print("📋 Projects in corpus:")
    
    import sqlite3
    with sqlite3.connect(updater.db_path) as conn:
        cursor = conn.execute("""
            SELECT project, COUNT(*) as file_count, SUM(chunk_count) as total_chunks
            FROM file_hashes 
            GROUP BY project
            ORDER BY total_chunks DESC
        """)
        
        for row in cursor.fetchall():
            project, file_count, chunk_count = row
            print(f"   📦 {project}: {file_count} files, {chunk_count} chunks")

def create_sample_config():
    """Create a sample configuration file."""
    sample_config = {
        "projects": [
            {
                "name": "my-web-app",
                "path": "/path/to/my-web-app"
            },
            {
                "name": "api-server", 
                "path": "/path/to/api-server"
            },
            {
                "name": "mobile-app",
                "path": "/path/to/mobile-app"
            }
        ]
    }
    
    config_path = "corpus_update_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
        
    print(f"✅ Created sample config: {config_path}")
    print("   Edit this file with your project paths, then run:")
    print(f"   python3 scripts/update_corpus.py bulk-update --config {config_path}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Manage incremental updates to the semantic search corpus"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show corpus statistics')
    
    # Add file command
    file_parser = subparsers.add_parser('add-file', help='Add single file')
    file_parser.add_argument('file', help='Path to code file')
    file_parser.add_argument('project', help='Project name')
    file_parser.add_argument('--language', help='Programming language (auto-detected if not specified)')
    
    # Add project command
    project_parser = subparsers.add_parser('add-project', help='Add entire project')
    project_parser.add_argument('name', help='Project name')
    project_parser.add_argument('path', help='Path to project directory')
    
    # Bulk update command
    bulk_parser = subparsers.add_parser('bulk-update', help='Bulk update from config file')
    bulk_parser.add_argument('--config', default='corpus_update_config.json', 
                            help='Config file path (default: corpus_update_config.json)')
    
    # Remove project command
    remove_parser = subparsers.add_parser('remove-project', help='Remove project from corpus')
    remove_parser.add_argument('name', help='Project name to remove')
    
    # List projects command
    list_parser = subparsers.add_parser('list-projects', help='List all projects')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create sample config file')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'create-config':
        create_sample_config()
        return
        
    # Initialize updater
    print("🔧 Initializing corpus updater...")
    updater = IncrementalUpdater()
    
    # Show initial stats
    if args.command != 'stats':
        print_stats(updater.get_corpus_stats())
        print()
    
    # Execute command
    if args.command == 'stats':
        print_stats(updater.get_corpus_stats())
        
    elif args.command == 'add-file':
        await add_single_file(updater, args)
        
    elif args.command == 'add-project':
        await add_project(updater, args)
        
    elif args.command == 'bulk-update':
        await bulk_update(updater, args)
        
    elif args.command == 'remove-project':
        await remove_project(updater, args)
        
    elif args.command == 'list-projects':
        await list_projects(updater, args)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())