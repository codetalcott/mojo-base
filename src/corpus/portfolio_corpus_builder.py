#!/usr/bin/env python3
"""
Portfolio Corpus Builder
Create comprehensive corpus from 40+ portfolio projects
Extends onedev vector data with additional project embeddings
"""

import os
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioCorpusBuilder:
    """Build comprehensive corpus from all portfolio projects."""
    
    def __init__(self):
        self.projects_root = "/Users/williamtalcott/projects"
        self.onedev_corpus_path = "<project-root>/data/real_vector_corpus.json"
        self.output_corpus_path = "<project-root>/data/portfolio_corpus.json"
        self.discovered_projects = []
        self.corpus_entries = []
        
    def discover_portfolio_projects(self) -> List[Dict]:
        """Discover all portfolio projects in the projects directory."""
        logger.info("🔍 Discovering portfolio projects")
        
        if not Path(self.projects_root).exists():
            raise FileNotFoundError(f"Projects root not found: {self.projects_root}")
        
        projects = []
        
        # Get all directories in projects root
        for item in Path(self.projects_root).iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                project_info = self._analyze_project(item)
                if project_info:
                    projects.append(project_info)
        
        self.discovered_projects = projects
        logger.info(f"📊 Discovered {len(projects)} portfolio projects")
        
        return projects
    
    def _analyze_project(self, project_path: Path) -> Optional[Dict]:
        """Analyze a single project to determine type and characteristics."""
        project_info = {
            "name": project_path.name,
            "path": str(project_path),
            "type": "unknown",
            "languages": [],
            "file_count": 0,
            "size_mb": 0,
            "has_code": False,
            "technologies": []
        }
        
        try:
            # Check for common files to determine project type
            common_files = list(project_path.glob("*"))
            file_names = [f.name for f in common_files if f.is_file()]
            
            # Determine project type
            if "package.json" in file_names:
                project_info["type"] = "nodejs"
                project_info["technologies"].append("Node.js")
            
            if "pyproject.toml" in file_names or "requirements.txt" in file_names:
                project_info["type"] = "python"
                project_info["technologies"].append("Python")
            
            if "Cargo.toml" in file_names:
                project_info["type"] = "rust"
                project_info["technologies"].append("Rust")
            
            if "go.mod" in file_names:
                project_info["type"] = "go"
                project_info["technologies"].append("Go")
            
            if "Dockerfile" in file_names:
                project_info["technologies"].append("Docker")
            
            if "README.md" in file_names:
                project_info["technologies"].append("Documentation")
            
            # Count code files and determine languages
            code_extensions = {
                '.py': 'python',
                '.js': 'javascript', 
                '.ts': 'typescript',
                '.tsx': 'typescript',
                '.jsx': 'javascript',
                '.rs': 'rust',
                '.go': 'go',
                '.mojo': 'mojo',
                '.java': 'java',
                '.c': 'c',
                '.cpp': 'cpp',
                '.h': 'c_header',
                '.hpp': 'cpp_header'
            }
            
            language_counts = {}
            total_files = 0
            total_size = 0
            
            for file_path in project_path.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    try:
                        size = file_path.stat().st_size
                        total_size += size
                        
                        ext = file_path.suffix.lower()
                        if ext in code_extensions:
                            lang = code_extensions[ext]
                            language_counts[lang] = language_counts.get(lang, 0) + 1
                            project_info["has_code"] = True
                    except (OSError, PermissionError):
                        continue
            
            project_info["file_count"] = total_files
            project_info["size_mb"] = total_size / (1024 * 1024)
            project_info["languages"] = list(language_counts.keys())
            
            # Skip projects that are too small or have no code
            if not project_info["has_code"] or project_info["file_count"] < 5:
                return None
            
            return project_info
            
        except Exception as e:
            logger.warning(f"Error analyzing project {project_path.name}: {e}")
            return None
    
    def load_existing_onedev_corpus(self) -> List[Dict]:
        """Load existing onedev corpus as base."""
        logger.info("📦 Loading existing onedev corpus")
        
        if not Path(self.onedev_corpus_path).exists():
            logger.warning(f"Onedev corpus not found: {self.onedev_corpus_path}")
            return []
        
        try:
            with open(self.onedev_corpus_path, 'r') as f:
                corpus_data = json.load(f)
            
            vectors = corpus_data.get("vectors", [])
            logger.info(f"✅ Loaded {len(vectors)} vectors from onedev corpus")
            
            # Convert to standard format
            standardized_vectors = []
            for vector in vectors:
                standardized_vector = {
                    "id": vector["id"],
                    "text": vector["text"],
                    "file_path": vector["file_path"],
                    "context_type": vector["context_type"],
                    "language": vector["language"],
                    "project": "onedev",
                    "source": "onedev_database",
                    "start_line": vector.get("start_line", 0),
                    "end_line": vector.get("end_line", 0),
                    "confidence": vector.get("confidence", 1.0),
                    "embedding": vector.get("embedding", []),
                    "metadata": {
                        "vector_dimensions": len(vector.get("embedding", [])),
                        "extraction_method": "onedev_context_db",
                        "quality_score": 100.0
                    }
                }
                standardized_vectors.append(standardized_vector)
            
            return standardized_vectors
            
        except Exception as e:
            logger.error(f"Error loading onedev corpus: {e}")
            return []
    
    def extract_code_snippets_from_project(self, project_info: Dict, max_snippets: int = 50) -> List[Dict]:
        """Extract code snippets from a project for corpus."""
        logger.info(f"📝 Extracting snippets from {project_info['name']}")
        
        project_path = Path(project_info["path"])
        snippets = []
        
        # File extensions to process
        code_extensions = {'.py', '.js', '.ts', '.tsx', '.jsx', '.mojo', '.go', '.rs'}
        
        try:
            code_files = []
            for ext in code_extensions:
                code_files.extend(list(project_path.rglob(f"*{ext}")))
            
            # Limit files to process
            code_files = code_files[:20]  # Process up to 20 files per project
            
            for file_path in code_files:
                try:
                    if file_path.stat().st_size > 100000:  # Skip very large files
                        continue
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract different types of snippets
                    file_snippets = self._extract_snippets_from_content(
                        content, file_path, project_info["name"]
                    )
                    snippets.extend(file_snippets)
                    
                    if len(snippets) >= max_snippets:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
                    continue
            
            logger.info(f"  📊 Extracted {len(snippets)} snippets from {project_info['name']}")
            return snippets[:max_snippets]  # Limit to max_snippets
            
        except Exception as e:
            logger.error(f"Error extracting from project {project_info['name']}: {e}")
            return []
    
    def _extract_snippets_from_content(self, content: str, file_path: Path, project_name: str) -> List[Dict]:
        """Extract meaningful code snippets from file content."""
        snippets = []
        lines = content.split('\n')
        
        # Extract functions and classes
        current_snippet = []
        snippet_start = 0
        in_function = False
        in_class = False
        brace_count = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Detect function/class starts
            if (stripped.startswith('function ') or 
                stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('fn ') or
                'function' in stripped):
                
                if current_snippet:
                    # Save previous snippet
                    snippet_text = '\n'.join(current_snippet)
                    if len(snippet_text.strip()) > 20:  # Minimum snippet size
                        snippet = self._create_snippet_entry(
                            snippet_text, file_path, project_name, 
                            snippet_start, i-1, "function"
                        )
                        snippets.append(snippet)
                
                # Start new snippet
                current_snippet = [line]
                snippet_start = i
                in_function = True
                brace_count = line.count('{') - line.count('}')
            
            elif in_function:
                current_snippet.append(line)
                brace_count += line.count('{') - line.count('}')
                
                # End of function (simplified detection)
                if brace_count <= 0 and len(current_snippet) > 1:
                    snippet_text = '\n'.join(current_snippet)
                    if len(snippet_text.strip()) > 20:
                        snippet = self._create_snippet_entry(
                            snippet_text, file_path, project_name,
                            snippet_start, i, "function"
                        )
                        snippets.append(snippet)
                    
                    current_snippet = []
                    in_function = False
                    brace_count = 0
        
        # Handle remaining snippet
        if current_snippet and len('\n'.join(current_snippet).strip()) > 20:
            snippet_text = '\n'.join(current_snippet)
            snippet = self._create_snippet_entry(
                snippet_text, file_path, project_name,
                snippet_start, len(lines)-1, "code_block"
            )
            snippets.append(snippet)
        
        # Also create a full-file snippet for smaller files
        if len(content) < 2000 and len(content.strip()) > 100:
            full_file_snippet = self._create_snippet_entry(
                content, file_path, project_name, 0, len(lines)-1, "full_file"
            )
            snippets.append(full_file_snippet)
        
        return snippets
    
    def _create_snippet_entry(self, text: str, file_path: Path, project_name: str, 
                            start_line: int, end_line: int, context_type: str) -> Dict:
        """Create a standardized snippet entry."""
        
        # Determine language from file extension
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.mojo': 'mojo',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        language = lang_map.get(file_path.suffix.lower(), 'unknown')
        
        # Create unique ID
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        snippet_id = f"{project_name}_{file_path.stem}_{start_line}_{content_hash}"
        
        return {
            "id": snippet_id,
            "text": text.strip(),
            "file_path": str(file_path.relative_to(Path("/Users/williamtalcott/projects"))),
            "context_type": context_type,
            "language": language,
            "project": project_name,
            "source": "portfolio_extraction",
            "start_line": start_line,
            "end_line": end_line,
            "confidence": 0.8,  # Lower than onedev vectors since not ML-generated
            "embedding": [],  # To be generated later
            "metadata": {
                "vector_dimensions": 0,  # To be set when embeddings generated
                "extraction_method": "static_analysis",
                "file_size": len(text),
                "project_type": "portfolio"
            }
        }
    
    def build_comprehensive_corpus(self) -> Dict:
        """Build comprehensive corpus from all sources."""
        logger.info("🚀 Building comprehensive portfolio corpus")
        
        # Step 1: Load existing onedev corpus
        onedev_vectors = self.load_existing_onedev_corpus()
        
        # Step 2: Discover portfolio projects
        projects = self.discover_portfolio_projects()
        
        # Step 3: Extract snippets from each project
        all_snippets = onedev_vectors.copy()
        
        for project in projects:
            if project["has_code"]:
                project_snippets = self.extract_code_snippets_from_project(project)
                all_snippets.extend(project_snippets)
        
        # Step 4: Create comprehensive corpus metadata
        languages = set()
        context_types = set()
        projects_included = set()
        
        for snippet in all_snippets:
            languages.add(snippet["language"])
            context_types.add(snippet["context_type"])
            projects_included.add(snippet["project"])
        
        corpus_metadata = {
            "creation_date": datetime.now().isoformat(),
            "total_vectors": len(all_snippets),
            "vector_dimensions": 128,  # Based on onedev vectors
            "corpus_version": "2.0_portfolio",
            "source_projects": len(projects_included),
            "languages": sorted(list(languages)),
            "context_types": sorted(list(context_types)),
            "projects_included": sorted(list(projects_included)),
            "onedev_vectors": len(onedev_vectors),
            "portfolio_vectors": len(all_snippets) - len(onedev_vectors),
            "extraction_methods": ["onedev_database", "static_analysis"],
            "quality_score": self._calculate_corpus_quality(all_snippets)
        }
        
        comprehensive_corpus = {
            "metadata": corpus_metadata,
            "vectors": all_snippets
        }
        
        # Step 5: Save corpus
        Path(self.output_corpus_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_corpus_path, 'w') as f:
            json.dump(comprehensive_corpus, f, indent=2)
        
        logger.info(f"✅ Comprehensive corpus saved to: {self.output_corpus_path}")
        
        return comprehensive_corpus
    
    def _calculate_corpus_quality(self, snippets: List[Dict]) -> float:
        """Calculate overall quality score for the corpus."""
        if not snippets:
            return 0.0
        
        quality_score = 0.0
        
        # Size score (40 points)
        if len(snippets) >= 2000:
            quality_score += 40
        elif len(snippets) >= 1000:
            quality_score += 30
        elif len(snippets) >= 500:
            quality_score += 20
        
        # Diversity score (30 points)
        languages = set(s["language"] for s in snippets)
        context_types = set(s["context_type"] for s in snippets)
        projects = set(s["project"] for s in snippets)
        
        diversity = len(languages) + len(context_types) + min(len(projects), 10)
        quality_score += min(diversity * 2, 30)
        
        # Quality score (30 points)
        avg_confidence = sum(s["confidence"] for s in snippets) / len(snippets)
        quality_score += avg_confidence * 30
        
        return min(quality_score, 100.0)
    
    def print_corpus_summary(self, corpus: Dict):
        """Print formatted summary of the built corpus."""
        metadata = corpus.get("metadata", {})
        
        print("\n" + "=" * 60)
        print("📋 PORTFOLIO CORPUS SUMMARY")
        print("=" * 60)
        
        print(f"📊 Total Vectors: {metadata.get('total_vectors'):,}")
        print(f"📏 Vector Dimensions: {metadata.get('vector_dimensions')}")
        print(f"🗂️ Source Projects: {metadata.get('source_projects')}")
        print(f"💬 Languages: {', '.join(metadata.get('languages', []))}")
        print(f"🏷️ Context Types: {', '.join(metadata.get('context_types', []))}")
        print(f"⭐ Quality Score: {metadata.get('quality_score', 0):.1f}/100")
        
        print(f"\n📦 Vector Sources:")
        print(f"  🧬 Onedev vectors: {metadata.get('onedev_vectors'):,}")
        print(f"  📁 Portfolio vectors: {metadata.get('portfolio_vectors'):,}")
        
        print(f"\n📁 Projects Included: {metadata.get('source_projects')}")
        for project in metadata.get('projects_included', [])[:10]:  # Show first 10
            print(f"  - {project}")
        if len(metadata.get('projects_included', [])) > 10:
            remaining = len(metadata.get('projects_included', [])) - 10
            print(f"  ... and {remaining} more projects")
        
        print("\n✅ Comprehensive portfolio corpus ready for Mojo integration!")

def main():
    """Main function to build portfolio corpus."""
    print("🚀 Portfolio Corpus Builder")
    print("==========================")
    print("Building comprehensive corpus from 40+ portfolio projects")
    print()
    
    builder = PortfolioCorpusBuilder()
    
    try:
        # Build comprehensive corpus
        corpus = builder.build_comprehensive_corpus()
        
        # Print summary
        builder.print_corpus_summary(corpus)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Corpus building failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)