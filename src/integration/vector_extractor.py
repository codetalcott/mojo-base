#!/usr/bin/env python3
"""
Vector Extractor and Validator
Extract and validate real vector embeddings from onedev database
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorExtractor:
    """Extract and validate vector embeddings from onedev database."""
    
    def __init__(self):
        self.context_db_path = "/Users/williamtalcott/projects/onedev/.onedev/context.db"
        self.extracted_vectors = []
        self.vector_metadata = {}
        self.validation_results = {}
        
    def extract_vector_embeddings(self) -> List[Dict]:
        """Extract all vector embeddings with metadata."""
        logger.info("🧬 Extracting vector embeddings from onedev database")
        
        if not Path(self.context_db_path).exists():
            raise FileNotFoundError(f"Context database not found: {self.context_db_path}")
        
        try:
            conn = sqlite3.connect(self.context_db_path)
            cursor = conn.cursor()
            
            # Extract all vectors with full metadata
            query = """
            SELECT 
                cv.id,
                cf.file_path,
                cv.context_type,
                cv.start_line,
                cv.end_line,
                cv.original_text,
                cv.code_snippet_hash,
                cv.confidence,
                cv.vector,
                cf.size,
                cf.language,
                cf.last_modified_at
            FROM code_vectors cv
            LEFT JOIN code_files cf ON cv.file_id = cf.id
            WHERE cv.vector IS NOT NULL
            ORDER BY cf.file_path, cv.start_line;
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            logger.info(f"📊 Found {len(rows)} vector embeddings")
            
            self.extracted_vectors = []
            for row in rows:
                try:
                    # Decode vector data
                    vector_blob = row[8]
                    vector_array = np.frombuffer(vector_blob, dtype=np.float32)
                    
                    vector_entry = {
                        "id": row[0],
                        "file_path": row[1],
                        "context_type": row[2],
                        "start_line": row[3],
                        "end_line": row[4],
                        "original_text": row[5],
                        "code_snippet_hash": row[6],
                        "confidence": row[7],
                        "vector_dimensions": len(vector_array),
                        "vector_magnitude": float(np.linalg.norm(vector_array)),
                        "vector_mean": float(np.mean(vector_array)),
                        "vector_std": float(np.std(vector_array)),
                        "vector_data": vector_array.tolist(),  # For JSON serialization
                        "file_size": row[9],
                        "language": row[10],
                        "last_modified": row[11]
                    }
                    
                    self.extracted_vectors.append(vector_entry)
                    
                except Exception as e:
                    logger.warning(f"Failed to decode vector for ID {row[0]}: {e}")
                    continue
            
            conn.close()
            
            logger.info(f"✅ Successfully extracted {len(self.extracted_vectors)} vectors")
            return self.extracted_vectors
            
        except Exception as e:
            logger.error(f"❌ Error extracting vectors: {e}")
            raise

    def validate_vector_quality(self) -> Dict:
        """Validate the quality and consistency of extracted vectors."""
        logger.info("🔬 Validating vector quality and consistency")
        
        if not self.extracted_vectors:
            raise ValueError("No vectors to validate. Run extract_vector_embeddings() first.")
        
        # Collect statistics
        dimensions = [v["vector_dimensions"] for v in self.extracted_vectors]
        magnitudes = [v["vector_magnitude"] for v in self.extracted_vectors]
        means = [v["vector_mean"] for v in self.extracted_vectors]
        stds = [v["vector_std"] for v in self.extracted_vectors]
        confidences = [v["confidence"] for v in self.extracted_vectors if v["confidence"] is not None]
        
        # Analyze by context type
        context_types = {}
        for vector in self.extracted_vectors:
            ctx_type = vector["context_type"]
            if ctx_type not in context_types:
                context_types[ctx_type] = []
            context_types[ctx_type].append(vector)
        
        # Analyze by language
        languages = {}
        for vector in self.extracted_vectors:
            lang = vector["language"] or "unknown"
            if lang not in languages:
                languages[lang] = []
            languages[lang].append(vector)
        
        validation_results = {
            "total_vectors": len(self.extracted_vectors),
            "dimension_analysis": {
                "unique_dimensions": list(set(dimensions)),
                "dimension_distribution": {dim: dimensions.count(dim) for dim in set(dimensions)},
                "consistent_dimensions": len(set(dimensions)) == 1
            },
            "magnitude_analysis": {
                "mean_magnitude": np.mean(magnitudes),
                "std_magnitude": np.std(magnitudes),
                "min_magnitude": np.min(magnitudes),
                "max_magnitude": np.max(magnitudes)
            },
            "confidence_analysis": {
                "mean_confidence": np.mean(confidences) if confidences else None,
                "std_confidence": np.std(confidences) if confidences else None,
                "min_confidence": np.min(confidences) if confidences else None,
                "max_confidence": np.max(confidences) if confidences else None,
                "vectors_with_confidence": len(confidences)
            },
            "context_type_analysis": {
                ctx_type: {
                    "count": len(vectors),
                    "percentage": (len(vectors) / len(self.extracted_vectors)) * 100
                }
                for ctx_type, vectors in context_types.items()
            },
            "language_analysis": {
                lang: {
                    "count": len(vectors),
                    "percentage": (len(vectors) / len(self.extracted_vectors)) * 100
                }
                for lang, vectors in languages.items()
            },
            "file_coverage": {
                "unique_files": len(set(v["file_path"] for v in self.extracted_vectors)),
                "vectors_per_file": len(self.extracted_vectors) / len(set(v["file_path"] for v in self.extracted_vectors))
            }
        }
        
        self.validation_results = validation_results
        
        # Log key findings
        logger.info(f"  📊 Total vectors: {validation_results['total_vectors']:,}")
        logger.info(f"  📏 Vector dimensions: {validation_results['dimension_analysis']['unique_dimensions']}")
        logger.info(f"  📁 Unique files: {validation_results['file_coverage']['unique_files']}")
        logger.info(f"  🏷️ Context types: {list(validation_results['context_type_analysis'].keys())}")
        logger.info(f"  💬 Languages: {list(validation_results['language_analysis'].keys())}")
        
        return validation_results

    def analyze_semantic_coherence(self, sample_size: int = 100) -> Dict:
        """Analyze semantic coherence of vectors through similarity analysis."""
        logger.info(f"🧠 Analyzing semantic coherence (sample size: {sample_size})")
        
        if not self.extracted_vectors:
            raise ValueError("No vectors to analyze. Run extract_vector_embeddings() first.")
        
        # Sample vectors for analysis
        import random
        sample_vectors = random.sample(self.extracted_vectors, min(sample_size, len(self.extracted_vectors)))
        
        # Convert to numpy arrays for analysis
        vectors_array = np.array([v["vector_data"] for v in sample_vectors])
        
        # Compute pairwise similarities
        similarities = []
        same_file_similarities = []
        same_type_similarities = []
        different_similarities = []
        
        for i in range(len(sample_vectors)):
            for j in range(i + 1, len(sample_vectors)):
                vec1, vec2 = vectors_array[i], vectors_array[j]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                similarities.append(similarity)
                
                # Categorize similarity
                v1, v2 = sample_vectors[i], sample_vectors[j]
                if v1["file_path"] == v2["file_path"]:
                    same_file_similarities.append(similarity)
                elif v1["context_type"] == v2["context_type"]:
                    same_type_similarities.append(similarity)
                else:
                    different_similarities.append(similarity)
        
        coherence_analysis = {
            "sample_size": len(sample_vectors),
            "total_comparisons": len(similarities),
            "overall_similarity": {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities))
            },
            "same_file_similarity": {
                "count": len(same_file_similarities),
                "mean": float(np.mean(same_file_similarities)) if same_file_similarities else None,
                "std": float(np.std(same_file_similarities)) if same_file_similarities else None
            },
            "same_type_similarity": {
                "count": len(same_type_similarities),
                "mean": float(np.mean(same_type_similarities)) if same_type_similarities else None,
                "std": float(np.std(same_type_similarities)) if same_type_similarities else None
            },
            "different_similarity": {
                "count": len(different_similarities),
                "mean": float(np.mean(different_similarities)) if different_similarities else None,
                "std": float(np.std(different_similarities)) if different_similarities else None
            }
        }
        
        logger.info(f"  📊 Overall similarity mean: {coherence_analysis['overall_similarity']['mean']:.3f}")
        logger.info(f"  📁 Same file similarity: {coherence_analysis['same_file_similarity']['mean']:.3f}" if same_file_similarities else "  📁 Same file similarity: N/A")
        logger.info(f"  🏷️ Same type similarity: {coherence_analysis['same_type_similarity']['mean']:.3f}" if same_type_similarities else "  🏷️ Same type similarity: N/A")
        logger.info(f"  🔀 Different similarity: {coherence_analysis['different_similarity']['mean']:.3f}" if different_similarities else "  🔀 Different similarity: N/A")
        
        return coherence_analysis

    def create_sample_corpus(self, output_path: str, sample_size: int = 1000) -> Dict:
        """Create a sample corpus file for Mojo integration testing."""
        logger.info(f"📝 Creating sample corpus (size: {sample_size})")
        
        if not self.extracted_vectors:
            raise ValueError("No vectors to sample. Run extract_vector_embeddings() first.")
        
        # Sample vectors
        import random
        sample_vectors = random.sample(self.extracted_vectors, min(sample_size, len(self.extracted_vectors)))
        
        # Create corpus format suitable for Mojo search engine
        corpus_data = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "source": "onedev_context_database",
                "total_vectors": len(sample_vectors),
                "vector_dimensions": sample_vectors[0]["vector_dimensions"] if sample_vectors else 0,
                "corpus_version": "1.0"
            },
            "vectors": []
        }
        
        for i, vector in enumerate(sample_vectors):
            corpus_entry = {
                "id": f"onedev_{vector['id']}",
                "text": vector["original_text"],
                "file_path": vector["file_path"],
                "context_type": vector["context_type"],
                "start_line": vector["start_line"],
                "end_line": vector["end_line"],
                "language": vector["language"],
                "embedding": vector["vector_data"],
                "confidence": vector["confidence"],
                "metadata": {
                    "file_size": vector["file_size"],
                    "last_modified": vector["last_modified"],
                    "snippet_hash": vector["code_snippet_hash"]
                }
            }
            corpus_data["vectors"].append(corpus_entry)
        
        # Save corpus
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(corpus_data, f, indent=2)
        
        logger.info(f"✅ Sample corpus saved to: {output_path}")
        
        return {
            "corpus_path": output_path,
            "sample_size": len(sample_vectors),
            "file_size_mb": Path(output_path).stat().st_size / (1024 * 1024)
        }

    def run_complete_extraction(self) -> Dict:
        """Run complete extraction and validation pipeline."""
        logger.info("🚀 Starting Complete Vector Extraction and Validation")
        logger.info("=" * 60)
        
        extraction_start = datetime.now()
        
        try:
            # Step 1: Extract vectors
            logger.info("\n📊 Step 1: Extracting Vector Embeddings")
            vectors = self.extract_vector_embeddings()
            
            # Step 2: Validate quality
            logger.info("\n🔬 Step 2: Validating Vector Quality")
            validation = self.validate_vector_quality()
            
            # Step 3: Analyze semantic coherence
            logger.info("\n🧠 Step 3: Analyzing Semantic Coherence")
            coherence = self.analyze_semantic_coherence()
            
            # Step 4: Create sample corpus
            logger.info("\n📝 Step 4: Creating Sample Corpus")
            corpus_path = "/Users/williamtalcott/projects/mojo-base/data/real_vector_corpus.json"
            corpus_info = self.create_sample_corpus(corpus_path, sample_size=1000)
            
            # Compile results
            results = {
                "extraction_timestamp": extraction_start.isoformat(),
                "extraction_duration_seconds": (datetime.now() - extraction_start).total_seconds(),
                "vectors_extracted": len(vectors),
                "validation_results": validation,
                "coherence_analysis": coherence,
                "sample_corpus": corpus_info,
                "extraction_summary": {
                    "status": "SUCCESS",
                    "vector_count": len(vectors),
                    "unique_files": validation["file_coverage"]["unique_files"],
                    "context_types": list(validation["context_type_analysis"].keys()),
                    "languages": list(validation["language_analysis"].keys()),
                    "vector_dimensions": validation["dimension_analysis"]["unique_dimensions"][0] if validation["dimension_analysis"]["unique_dimensions"] else None,
                    "quality_score": self._calculate_quality_score(validation, coherence)
                }
            }
            
            # Save detailed results
            results_path = "/Users/williamtalcott/projects/mojo-base/analysis/vector_extraction_results.json"
            Path(results_path).parent.mkdir(exist_ok=True)
            
            # Save without vector data for readability
            results_summary = {k: v for k, v in results.items()}
            with open(results_path, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            logger.info(f"\n✅ Extraction complete! Results saved to: {results_path}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            raise

    def _calculate_quality_score(self, validation: Dict, coherence: Dict) -> float:
        """Calculate overall quality score for the vector dataset."""
        score = 0.0
        
        # Dimension consistency (20 points)
        if validation["dimension_analysis"]["consistent_dimensions"]:
            score += 20
        
        # Vector count (20 points)
        vector_count = validation["total_vectors"]
        if vector_count >= 2000:
            score += 20
        elif vector_count >= 1000:
            score += 15
        elif vector_count >= 500:
            score += 10
        
        # Confidence scores (20 points)
        conf_analysis = validation["confidence_analysis"]
        if conf_analysis["mean_confidence"] and conf_analysis["mean_confidence"] > 0.8:
            score += 20
        elif conf_analysis["mean_confidence"] and conf_analysis["mean_confidence"] > 0.6:
            score += 15
        
        # Semantic coherence (20 points)
        overall_sim = coherence["overall_similarity"]["mean"]
        if 0.1 <= overall_sim <= 0.5:  # Good range for semantic diversity
            score += 20
        elif 0.05 <= overall_sim <= 0.6:
            score += 15
        
        # Coverage diversity (20 points)
        lang_count = len(validation["language_analysis"])
        type_count = len(validation["context_type_analysis"])
        if lang_count >= 3 and type_count >= 3:
            score += 20
        elif lang_count >= 2 and type_count >= 2:
            score += 15
        
        return score

    def print_extraction_summary(self, results: Dict):
        """Print formatted summary of extraction results."""
        summary = results.get("extraction_summary", {})
        
        print("\n" + "=" * 60)
        print("📋 VECTOR EXTRACTION SUMMARY")
        print("=" * 60)
        
        print(f"🔍 Extraction Status: {summary.get('status')}")
        print(f"🧬 Vectors Extracted: {summary.get('vector_count'):,}")
        print(f"📁 Unique Files: {summary.get('unique_files')}")
        print(f"📏 Vector Dimensions: {summary.get('vector_dimensions')}")
        print(f"🏷️ Context Types: {', '.join(summary.get('context_types', []))}")
        print(f"💬 Languages: {', '.join(summary.get('languages', []))}")
        print(f"⭐ Quality Score: {summary.get('quality_score'):.1f}/100")
        
        corpus_info = results.get("sample_corpus", {})
        print(f"\n📝 Sample Corpus Created:")
        print(f"  📍 Path: {corpus_info.get('corpus_path')}")
        print(f"  📊 Size: {corpus_info.get('sample_size')} vectors")
        print(f"  💾 File Size: {corpus_info.get('file_size_mb', 0):.2f} MB")
        
        print("\n✅ Vector extraction complete - Ready for Mojo integration!")

def main():
    """Main function to run vector extraction."""
    print("🚀 Vector Extraction and Validation for Mojo Integration")
    print("=======================================================")
    print("Extracting real vector embeddings from onedev database")
    print()
    
    extractor = VectorExtractor()
    
    try:
        # Run complete extraction
        results = extractor.run_complete_extraction()
        
        # Print summary
        extractor.print_extraction_summary(results)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)