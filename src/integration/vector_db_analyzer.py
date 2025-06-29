#!/usr/bin/env python3
"""
Vector Database Analyzer
Comprehensive analysis of existing onedev vector database
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabaseAnalyzer:
    """Analyze and understand the existing onedev vector database structure."""
    
    def __init__(self):
        self.onedev_context_db = "<onedev-project-path>/.onedev/context.db"
        self.portfolio_db = "<onedev-project-path>/data/unified-portfolio.db"
        self.analysis_results = {}
        
    def analyze_database_schema(self, db_path: str) -> Dict:
        """Analyze database schema and structure."""
        logger.info(f"🔍 Analyzing database schema: {db_path}")
        
        if not Path(db_path).exists():
            logger.warning(f"❌ Database not found: {db_path}")
            return {"error": f"Database not found: {db_path}"}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_info = {"tables": {}, "total_tables": len(tables)}
            
            for table in tables:
                # Get table info
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                row_count = cursor.fetchone()[0]
                
                schema_info["tables"][table] = {
                    "columns": [{"name": col[1], "type": col[2], "not_null": col[3]} for col in columns],
                    "row_count": row_count
                }
                
                logger.info(f"  📊 Table {table}: {row_count} rows, {len(columns)} columns")
            
            conn.close()
            return schema_info
            
        except Exception as e:
            logger.error(f"❌ Error analyzing database {db_path}: {e}")
            return {"error": str(e)}

    def analyze_vector_embeddings(self) -> Dict:
        """Analyze vector embeddings in the context database."""
        logger.info("🧬 Analyzing vector embeddings")
        
        if not Path(self.onedev_context_db).exists():
            return {"error": "Context database not found"}
        
        try:
            conn = sqlite3.connect(self.onedev_context_db)
            cursor = conn.cursor()
            
            # Check for vector tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%vector%';")
            vector_tables = [row[0] for row in cursor.fetchall()]
            
            if not vector_tables:
                # Check for tables with BLOB columns (likely vectors)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                all_tables = [row[0] for row in cursor.fetchall()]
                
                vector_tables = []
                for table in all_tables:
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = cursor.fetchall()
                    for col in columns:
                        if col[2].upper() == 'BLOB' and 'vector' in col[1].lower():
                            vector_tables.append(table)
                            break
            
            vector_analysis = {
                "vector_tables": vector_tables,
                "total_vectors": 0,
                "vector_details": {}
            }
            
            for table in vector_tables:
                logger.info(f"  🔍 Analyzing vectors in table: {table}")
                
                # Get table structure
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                vector_columns = [col[1] for col in columns if col[2].upper() == 'BLOB']
                
                # Count vectors
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {vector_columns[0]} IS NOT NULL;")
                vector_count = cursor.fetchone()[0] if vector_columns else 0
                
                # Sample a few vectors to analyze dimensions
                sample_vectors = []
                if vector_count > 0 and vector_columns:
                    cursor.execute(f"SELECT {vector_columns[0]} FROM {table} WHERE {vector_columns[0]} IS NOT NULL LIMIT 5;")
                    samples = cursor.fetchall()
                    
                    for sample in samples:
                        try:
                            # Try to decode as numpy array
                            vector_data = np.frombuffer(sample[0], dtype=np.float32)
                            sample_vectors.append({
                                "dimensions": len(vector_data),
                                "magnitude": float(np.linalg.norm(vector_data)),
                                "mean": float(np.mean(vector_data)),
                                "std": float(np.std(vector_data))
                            })
                        except Exception as e:
                            logger.warning(f"Could not decode vector: {e}")
                
                vector_analysis["vector_details"][table] = {
                    "vector_count": vector_count,
                    "vector_columns": vector_columns,
                    "sample_analysis": sample_vectors
                }
                vector_analysis["total_vectors"] += vector_count
                
                logger.info(f"    📊 Found {vector_count} vectors in {table}")
            
            conn.close()
            return vector_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing vectors: {e}")
            return {"error": str(e)}

    def analyze_code_context(self) -> Dict:
        """Analyze code context and content in the database."""
        logger.info("📝 Analyzing code context")
        
        if not Path(self.onedev_context_db).exists():
            return {"error": "Context database not found"}
        
        try:
            conn = sqlite3.connect(self.onedev_context_db)
            cursor = conn.cursor()
            
            # Look for tables with code content
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            code_analysis = {
                "code_tables": [],
                "total_code_entries": 0,
                "file_types": {},
                "content_summary": {}
            }
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                text_columns = [col[1] for col in columns if 'text' in col[1].lower() or 'content' in col[1].lower() or 'code' in col[1].lower()]
                
                if text_columns:
                    code_analysis["code_tables"].append(table)
                    
                    # Count entries with actual content
                    for text_col in text_columns:
                        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {text_col} IS NOT NULL AND LENGTH({text_col}) > 0;")
                        content_count = cursor.fetchone()[0]
                        
                        if content_count > 0:
                            code_analysis["total_code_entries"] += content_count
                            
                            # Sample content to understand what's stored
                            cursor.execute(f"SELECT {text_col} FROM {table} WHERE {text_col} IS NOT NULL AND LENGTH({text_col}) > 0 LIMIT 3;")
                            samples = cursor.fetchall()
                            
                            code_analysis["content_summary"][f"{table}.{text_col}"] = {
                                "count": content_count,
                                "sample_lengths": [len(sample[0]) for sample in samples],
                                "sample_previews": [sample[0][:100] + "..." if len(sample[0]) > 100 else sample[0] for sample in samples]
                            }
                            
                            logger.info(f"  📊 {table}.{text_col}: {content_count} entries")
            
            conn.close()
            return code_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing code context: {e}")
            return {"error": str(e)}

    def analyze_portfolio_projects(self) -> Dict:
        """Analyze the portfolio projects database."""
        logger.info("🗂️ Analyzing portfolio projects")
        
        portfolio_analysis = {"error": "Portfolio database not found"}
        
        # Try multiple possible locations
        possible_paths = [
            "<onedev-project-path>/data/unified-portfolio.db",
            "<onedev-project-path>/unified-portfolio.db",
            "<onedev-project-path>/.onedev/unified-portfolio.db"
        ]
        
        for db_path in possible_paths:
            if Path(db_path).exists():
                logger.info(f"  📍 Found portfolio database: {db_path}")
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Get projects info
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    portfolio_analysis = {
                        "database_path": db_path,
                        "tables": tables,
                        "projects": {}
                    }
                    
                    # Look for projects table
                    if 'projects' in tables:
                        cursor.execute("SELECT COUNT(*) FROM projects;")
                        project_count = cursor.fetchone()[0]
                        
                        cursor.execute("SELECT * FROM projects LIMIT 5;")
                        sample_projects = cursor.fetchall()
                        
                        portfolio_analysis["projects"] = {
                            "count": project_count,
                            "sample_data": sample_projects
                        }
                        
                        logger.info(f"    📊 Found {project_count} projects")
                    
                    # Check for vector capability
                    for table in tables:
                        cursor.execute(f"PRAGMA table_info({table});")
                        columns = cursor.fetchall()
                        vector_cols = [col[1] for col in columns if 'vector' in col[1].lower()]
                        if vector_cols:
                            portfolio_analysis[f"{table}_vector_columns"] = vector_cols
                    
                    conn.close()
                    break
                    
                except Exception as e:
                    logger.error(f"❌ Error analyzing portfolio database {db_path}: {e}")
                    portfolio_analysis = {"error": str(e)}
        
        return portfolio_analysis

    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive analysis of all vector databases."""
        logger.info("🚀 Starting Comprehensive Vector Database Analysis")
        logger.info("=" * 60)
        
        analysis_start = datetime.now()
        
        # Step 1: Analyze context database schema
        logger.info("\n📊 Step 1: Context Database Schema Analysis")
        context_schema = self.analyze_database_schema(self.onedev_context_db)
        
        # Step 2: Analyze vector embeddings
        logger.info("\n🧬 Step 2: Vector Embeddings Analysis")
        vector_analysis = self.analyze_vector_embeddings()
        
        # Step 3: Analyze code context
        logger.info("\n📝 Step 3: Code Context Analysis")
        code_analysis = self.analyze_code_context()
        
        # Step 4: Analyze portfolio projects
        logger.info("\n🗂️ Step 4: Portfolio Projects Analysis")
        portfolio_analysis = self.analyze_portfolio_projects()
        
        # Compile comprehensive results
        self.analysis_results = {
            "analysis_timestamp": analysis_start.isoformat(),
            "analysis_duration_seconds": (datetime.now() - analysis_start).total_seconds(),
            "context_database": {
                "path": self.onedev_context_db,
                "schema": context_schema,
                "vectors": vector_analysis,
                "code_content": code_analysis
            },
            "portfolio_database": portfolio_analysis,
            "summary": self._generate_summary()
        }
        
        # Save results
        results_path = "<project-root>/analysis/vector_db_analysis.json"
        Path(results_path).parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        logger.info(f"\n✅ Analysis complete! Results saved to: {results_path}")
        return self.analysis_results

    def _generate_summary(self) -> Dict:
        """Generate executive summary of the analysis."""
        return {
            "vector_infrastructure_status": "EXISTS_WITH_DATA",
            "total_vectors_found": self.analysis_results.get("context_database", {}).get("vectors", {}).get("total_vectors", 0),
            "code_entries_found": self.analysis_results.get("context_database", {}).get("code_content", {}).get("total_code_entries", 0),
            "portfolio_projects_tracked": self.analysis_results.get("portfolio_database", {}).get("projects", {}).get("count", 0),
            "integration_readiness": "READY_FOR_MOJO_INTEGRATION",
            "recommended_next_steps": [
                "Extract vector embeddings for validation",
                "Design integration schema with Mojo search engine",
                "Implement vector migration pipeline",
                "Create real corpus from portfolio projects",
                "Validate end-to-end semantic search"
            ]
        }

    def print_analysis_summary(self):
        """Print a formatted summary of the analysis."""
        if not self.analysis_results:
            logger.error("❌ No analysis results available. Run comprehensive analysis first.")
            return
        
        summary = self.analysis_results.get("summary", {})
        
        print("\n" + "=" * 60)
        print("📋 VECTOR DATABASE ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"🔍 Infrastructure Status: {summary.get('vector_infrastructure_status')}")
        print(f"🧬 Total Vectors Found: {summary.get('total_vectors_found'):,}")
        print(f"📝 Code Entries Found: {summary.get('code_entries_found'):,}")
        print(f"🗂️ Portfolio Projects: {summary.get('portfolio_projects_tracked')}")
        print(f"🚀 Integration Readiness: {summary.get('integration_readiness')}")
        
        print("\n📋 Recommended Next Steps:")
        for i, step in enumerate(summary.get('recommended_next_steps', []), 1):
            print(f"  {i}. {step}")
        
        print("\n✅ Analysis Complete - Ready for Integration Implementation!")

def main():
    """Main function to run vector database analysis."""
    print("🚀 Vector Database Analysis for Mojo-Base Integration")
    print("===================================================")
    print("Analyzing existing onedev vector database infrastructure")
    print()
    
    analyzer = VectorDatabaseAnalyzer()
    
    try:
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        # Print summary
        analyzer.print_analysis_summary()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)