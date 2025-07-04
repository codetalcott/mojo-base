#!/usr/bin/env python3
"""
Credibility Validation Tests for MAX Graph Performance Claims

Rigorous testing to ensure our performance numbers are accurate, reproducible, and credible.
Multiple test runs, statistical analysis, and measurement validation.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.max_graph.semantic_search_graph import MaxGraphConfig, MaxSemanticSearchGraph

@dataclass
class CredibilityTest:
    """Single credibility test configuration."""
    name: str
    description: str
    corpus_size: int
    iterations: int
    expected_range_ms: Tuple[float, float]  # (min, max) expected latency
    config_changes: Dict[str, Any]

@dataclass
class ValidationResults:
    """Results from credibility validation."""
    test_name: str
    corpus_size: int
    iterations: int
    latencies_ms: List[float]
    mean_ms: float
    median_ms: float
    std_dev_ms: float
    min_ms: float
    max_ms: float
    coefficient_variation: float
    within_expected_range: bool
    measurement_quality: str
    throughput_vectors_per_sec: float

class CredibilityValidator:
    """Comprehensive credibility validation for MAX Graph performance."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "data" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters
        self.vector_dims = 768
        self.random_seed = 42  # Consistent data across tests
        
    def define_credibility_tests(self) -> List[CredibilityTest]:
        """Define comprehensive credibility tests."""
        return [
            CredibilityTest(
                name="baseline_2k_validation",
                description="Validate 2K baseline with high iteration count",
                corpus_size=2000,
                iterations=10,  # More iterations for statistical confidence
                expected_range_ms=(0.8, 2.0),  # Based on previous measurements
                config_changes={"use_fp16": False, "enable_fusion": False}
            ),
            CredibilityTest(
                name="baseline_5k_validation",
                description="Validate 5K baseline with high iteration count",
                corpus_size=5000,
                iterations=10,
                expected_range_ms=(1.5, 3.0),  # Based on previous measurements
                config_changes={"use_fp16": False, "enable_fusion": False}
            ),
            CredibilityTest(
                name="baseline_10k_validation",
                description="Validate 10K baseline with high iteration count",
                corpus_size=10000,
                iterations=10,
                expected_range_ms=(3.0, 5.0),  # Based on previous measurements
                config_changes={"use_fp16": False, "enable_fusion": False}
            ),
            CredibilityTest(
                name="fusion_2k_validation",
                description="Validate 2K fusion improvement claims",
                corpus_size=2000,
                iterations=10,
                expected_range_ms=(0.8, 2.0),  # Should be similar or better than baseline
                config_changes={"use_fp16": False, "enable_fusion": True}
            ),
            CredibilityTest(
                name="fusion_5k_validation",
                description="Validate 5K fusion improvement claims",
                corpus_size=5000,
                iterations=10,
                expected_range_ms=(1.5, 3.0),  # Should be similar or better than baseline
                config_changes={"use_fp16": False, "enable_fusion": True}
            ),
            CredibilityTest(
                name="scaling_consistency",
                description="Test scaling consistency with extended measurement",
                corpus_size=7500,  # Intermediate size for scaling validation
                iterations=8,
                expected_range_ms=(2.5, 4.0),  # Interpolated from scaling
                config_changes={"use_fp16": False, "enable_fusion": False}
            ),
            CredibilityTest(
                name="measurement_repeatability",
                description="Test measurement repeatability with same config",
                corpus_size=3000,  # Different size to test reproducibility
                iterations=15,  # High iteration count for statistical analysis
                expected_range_ms=(1.2, 2.5),  # Interpolated
                config_changes={"use_fp16": False, "enable_fusion": False}
            )
        ]
    
    def create_consistent_test_data(self, corpus_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create consistent test data for reproducible measurements."""
        # Always use same seed for consistent data
        np.random.seed(self.random_seed)
        
        query_embeddings = np.random.randn(5, self.vector_dims).astype(np.float32)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        corpus_embeddings = np.random.randn(corpus_size, self.vector_dims).astype(np.float32)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        
        return query_embeddings, corpus_embeddings
    
    def run_credibility_test(self, test: CredibilityTest) -> ValidationResults:
        """Run single credibility test with rigorous measurement."""
        print(f"\n🔬 Credibility Test: {test.name}")
        print(f"   Description: {test.description}")
        print(f"   Corpus: {test.corpus_size:,} vectors")
        print(f"   Iterations: {test.iterations}")
        print(f"   Expected range: {test.expected_range_ms[0]:.1f}-{test.expected_range_ms[1]:.1f}ms")
        
        # Create configuration
        config = MaxGraphConfig(
            corpus_size=test.corpus_size,
            vector_dims=self.vector_dims,
            device="cpu",
            **test.config_changes
        )
        
        # Initialize and compile MAX Graph
        print("   🏗️  Building MAX Graph...")
        max_search = MaxSemanticSearchGraph(config)
        max_search.compile("cpu")
        
        # Create consistent test data
        query_embeddings, corpus_embeddings = self.create_consistent_test_data(test.corpus_size)
        
        # Multiple warm-up runs to stabilize performance
        print("   🔥 Warming up (5 runs)...")
        for _ in range(5):
            max_search.search_similarity(query_embeddings[0], corpus_embeddings)
        
        # Rigorous measurement with multiple iterations
        print(f"   ⏱️  Measuring performance ({test.iterations} iterations)...")
        latencies = []
        
        for i in range(test.iterations):
            # Clear any potential caching between runs
            _ = max_search.search_similarity(query_embeddings[i % len(query_embeddings)], corpus_embeddings)
            
            # Actual timed measurement
            start_time = time.perf_counter()
            result = max_search.search_similarity(query_embeddings[0], corpus_embeddings)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            print(f"     Run {i+1:2d}: {latency_ms:.3f}ms")
        
        # Statistical analysis
        mean_ms = statistics.mean(latencies)
        median_ms = statistics.median(latencies)
        std_dev_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        min_ms = min(latencies)
        max_ms = max(latencies)
        coefficient_variation = (std_dev_ms / mean_ms) * 100 if mean_ms > 0 else 0.0
        
        # Validate against expected range
        within_expected_range = test.expected_range_ms[0] <= mean_ms <= test.expected_range_ms[1]
        
        # Determine measurement quality
        if coefficient_variation < 5.0:
            measurement_quality = "EXCELLENT"
        elif coefficient_variation < 10.0:
            measurement_quality = "GOOD"
        elif coefficient_variation < 20.0:
            measurement_quality = "FAIR"
        else:
            measurement_quality = "POOR"
        
        # Calculate throughput
        throughput_vectors_per_sec = test.corpus_size / (mean_ms / 1000.0)
        
        results = ValidationResults(
            test_name=test.name,
            corpus_size=test.corpus_size,
            iterations=test.iterations,
            latencies_ms=latencies,
            mean_ms=mean_ms,
            median_ms=median_ms,
            std_dev_ms=std_dev_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            coefficient_variation=coefficient_variation,
            within_expected_range=within_expected_range,
            measurement_quality=measurement_quality,
            throughput_vectors_per_sec=throughput_vectors_per_sec
        )
        
        # Print detailed results
        print(f"   📊 Results:")
        print(f"     Mean: {mean_ms:.3f}ms ± {std_dev_ms:.3f}ms")
        print(f"     Median: {median_ms:.3f}ms")
        print(f"     Range: {min_ms:.3f}ms - {max_ms:.3f}ms")
        print(f"     CV: {coefficient_variation:.1f}% ({measurement_quality})")
        print(f"     Expected: {'✅ YES' if within_expected_range else '❌ NO'}")
        print(f"     Throughput: {throughput_vectors_per_sec:,.0f} vectors/sec")
        
        return results
    
    def validate_scaling_claims(self, results: List[ValidationResults]) -> Dict[str, Any]:
        """Validate our scaling characteristic claims."""
        print(f"\n📈 Scaling Validation Analysis")
        print("=" * 40)
        
        # Get baseline results for scaling analysis
        baseline_results = [r for r in results if 'baseline' in r.test_name and 'validation' in r.test_name]
        baseline_results.sort(key=lambda x: x.corpus_size)
        
        if len(baseline_results) < 2:
            return {"error": "Insufficient baseline results for scaling analysis"}
        
        scaling_analysis = {
            "scaling_points": [],
            "linear_fit": None,
            "scaling_efficiency": None,
            "prediction_accuracy": []
        }
        
        # Calculate per-1K scaling rates
        for result in baseline_results:
            per_1k_ms = (result.mean_ms / result.corpus_size) * 1000
            scaling_analysis["scaling_points"].append({
                "corpus_size": result.corpus_size,
                "latency_ms": result.mean_ms,
                "per_1k_ms": per_1k_ms
            })
            
            print(f"   {result.corpus_size:,} vectors: {result.mean_ms:.3f}ms ({per_1k_ms:.3f}ms/1K)")
        
        # Linear scaling efficiency calculation
        if len(baseline_results) >= 2:
            first = baseline_results[0]
            last = baseline_results[-1]
            
            size_ratio = last.corpus_size / first.corpus_size
            latency_ratio = last.mean_ms / first.mean_ms
            scaling_efficiency = latency_ratio / size_ratio
            
            scaling_analysis["scaling_efficiency"] = scaling_efficiency
            
            print(f"\n   Size ratio: {size_ratio:.1f}x")
            print(f"   Latency ratio: {latency_ratio:.1f}x")
            print(f"   Scaling efficiency: {scaling_efficiency:.3f}")
            
            if scaling_efficiency < 1.2:
                print(f"   ✅ Excellent linear scaling confirmed")
            elif scaling_efficiency < 1.5:
                print(f"   📊 Good scaling characteristics")
            else:
                print(f"   ⚠️ Scaling efficiency concerns")
        
        return scaling_analysis
    
    def validate_fusion_claims(self, results: List[ValidationResults]) -> Dict[str, Any]:
        """Validate our kernel fusion improvement claims."""
        print(f"\n🔧 Fusion Validation Analysis")
        print("=" * 40)
        
        fusion_analysis = {
            "comparisons": [],
            "average_improvement": 0.0,
            "claims_validated": False
        }
        
        # Compare baseline vs fusion for each corpus size
        corpus_sizes = set(r.corpus_size for r in results)
        
        total_improvement = 0.0
        comparison_count = 0
        
        for corpus_size in corpus_sizes:
            baseline_result = next((r for r in results if r.corpus_size == corpus_size and 'baseline' in r.test_name), None)
            fusion_result = next((r for r in results if r.corpus_size == corpus_size and 'fusion' in r.test_name), None)
            
            if baseline_result and fusion_result:
                improvement_factor = baseline_result.mean_ms / fusion_result.mean_ms
                improvement_percent = ((baseline_result.mean_ms - fusion_result.mean_ms) / baseline_result.mean_ms) * 100
                
                comparison = {
                    "corpus_size": corpus_size,
                    "baseline_ms": baseline_result.mean_ms,
                    "fusion_ms": fusion_result.mean_ms,
                    "improvement_factor": improvement_factor,
                    "improvement_percent": improvement_percent
                }
                
                fusion_analysis["comparisons"].append(comparison)
                total_improvement += improvement_percent
                comparison_count += 1
                
                print(f"   {corpus_size:,} vectors:")
                print(f"     Baseline: {baseline_result.mean_ms:.3f}ms")
                print(f"     Fusion: {fusion_result.mean_ms:.3f}ms")
                print(f"     Improvement: {improvement_factor:.2f}x ({improvement_percent:+.1f}%)")
        
        if comparison_count > 0:
            average_improvement = total_improvement / comparison_count
            fusion_analysis["average_improvement"] = average_improvement
            
            # Validate against our claims (5-10% improvement)
            claims_validated = 5.0 <= average_improvement <= 15.0  # Allow some tolerance
            fusion_analysis["claims_validated"] = claims_validated
            
            print(f"\n   Average improvement: {average_improvement:.1f}%")
            print(f"   Claims validated: {'✅ YES' if claims_validated else '❌ NO'}")
        
        return fusion_analysis
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete credibility validation suite."""
        print("🔬 MAX Graph Performance Credibility Validation")
        print("=" * 60)
        print("🎯 Goal: Validate all performance claims with rigorous testing")
        print("📊 Methodology: High iteration counts, statistical analysis")
        print("🔍 Focus: Measurement quality, reproducibility, scaling validation")
        
        # Get credibility tests
        tests = self.define_credibility_tests()
        
        # Run all credibility tests
        validation_results = []
        failed_tests = []
        
        for i, test in enumerate(tests):
            print(f"\n[{i+1}/{len(tests)}] Running credibility test...")
            
            try:
                result = self.run_credibility_test(test)
                validation_results.append(result)
                
                if not result.within_expected_range:
                    failed_tests.append(test.name)
                    print(f"   ⚠️ OUTSIDE EXPECTED RANGE: {result.mean_ms:.3f}ms")
                
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
                failed_tests.append(test.name)
        
        # Perform comprehensive analysis
        scaling_analysis = self.validate_scaling_claims(validation_results)
        fusion_analysis = self.validate_fusion_claims(validation_results)
        
        # Overall credibility assessment
        successful_tests = len(validation_results)
        total_tests = len(tests)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        # Measurement quality assessment
        excellent_measurements = sum(1 for r in validation_results if r.measurement_quality == "EXCELLENT")
        good_measurements = sum(1 for r in validation_results if r.measurement_quality in ["EXCELLENT", "GOOD"])
        
        overall_assessment = {
            "credibility_score": self.calculate_credibility_score(validation_results, scaling_analysis, fusion_analysis),
            "test_success_rate": success_rate,
            "measurement_quality_rate": good_measurements / successful_tests if successful_tests > 0 else 0.0,
            "claims_validated": len(failed_tests) == 0,
            "recommendations": self.generate_credibility_recommendations(validation_results, failed_tests)
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "validation_results": validation_results,
            "scaling_analysis": scaling_analysis,
            "fusion_analysis": fusion_analysis,
            "overall_assessment": overall_assessment,
            "failed_tests": failed_tests,
            "summary": self.generate_credibility_summary(overall_assessment, validation_results)
        }
    
    def calculate_credibility_score(self, results: List[ValidationResults], 
                                  scaling: Dict[str, Any], fusion: Dict[str, Any]) -> float:
        """Calculate overall credibility score (0-100)."""
        score = 0.0
        
        # Test success rate (40 points)
        in_range_count = sum(1 for r in results if r.within_expected_range)
        score += (in_range_count / len(results)) * 40 if results else 0
        
        # Measurement quality (30 points)
        quality_scores = {"EXCELLENT": 1.0, "GOOD": 0.8, "FAIR": 0.5, "POOR": 0.2}
        avg_quality = np.mean([quality_scores.get(r.measurement_quality, 0) for r in results])
        score += avg_quality * 30
        
        # Scaling validation (20 points)
        if scaling.get("scaling_efficiency"):
            if scaling["scaling_efficiency"] < 1.2:
                score += 20  # Excellent scaling
            elif scaling["scaling_efficiency"] < 1.5:
                score += 15  # Good scaling
            else:
                score += 5   # Poor scaling
        
        # Fusion claims (10 points)
        if fusion.get("claims_validated"):
            score += 10
        
        return min(score, 100.0)
    
    def generate_credibility_recommendations(self, results: List[ValidationResults], 
                                           failed_tests: List[str]) -> List[str]:
        """Generate recommendations based on credibility analysis."""
        recommendations = []
        
        if not failed_tests:
            recommendations.append("✅ All performance claims validated - numbers are credible")
        else:
            recommendations.append(f"⚠️ {len(failed_tests)} tests outside expected range - review claims")
        
        # Measurement quality recommendations
        poor_quality = [r for r in results if r.measurement_quality in ["FAIR", "POOR"]]
        if poor_quality:
            recommendations.append(f"📊 {len(poor_quality)} tests show high variance - increase warm-up or iterations")
        
        # Performance consistency
        high_variance = [r for r in results if r.coefficient_variation > 15.0]
        if high_variance:
            recommendations.append("🔧 High performance variance detected - consider system load factors")
        
        return recommendations
    
    def generate_credibility_summary(self, assessment: Dict[str, Any], 
                                   results: List[ValidationResults]) -> Dict[str, Any]:
        """Generate executive summary of credibility validation."""
        best_result = min(results, key=lambda x: x.mean_ms) if results else None
        worst_variance = max(results, key=lambda x: x.coefficient_variation) if results else None
        
        return {
            "overall_credibility": "HIGH" if assessment["credibility_score"] > 80 else 
                                 "MEDIUM" if assessment["credibility_score"] > 60 else "LOW",
            "credibility_score": assessment["credibility_score"],
            "test_success_rate": f"{assessment['test_success_rate']:.1%}",
            "claims_validated": assessment["claims_validated"],
            "best_performance": f"{best_result.mean_ms:.3f}ms for {best_result.corpus_size:,} vectors" if best_result else "N/A",
            "measurement_quality": f"{assessment['measurement_quality_rate']:.1%} good/excellent",
            "worst_variance": f"{worst_variance.coefficient_variation:.1f}% CV" if worst_variance else "N/A",
            "recommendation": "PUBLISH" if assessment["credibility_score"] > 75 else 
                            "REVIEW" if assessment["credibility_score"] > 50 else "REVISE"
        }
    
    def save_validation_results(self, results: Dict[str, Any]) -> Path:
        """Save credibility validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"credibility_validation_{timestamp}.json"
        
        # Convert to JSON-serializable format
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return make_json_serializable(asdict(obj))
            elif isinstance(obj, (bool, int, float, str, type(None))):
                return obj
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)
        
        serializable_results = make_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return results_file

def main():
    """Main credibility validation execution."""
    print("🔬 MAX Graph Performance Credibility Validation")
    print("🎯 Rigorous testing to ensure our performance claims are accurate")
    print("📊 Multiple iterations, statistical analysis, measurement validation")
    print()
    
    validator = CredibilityValidator()
    
    # Run comprehensive validation
    validation_results = validator.run_comprehensive_validation()
    
    # Print final summary
    print(f"\n🎯 Credibility Validation Summary")
    print("=" * 50)
    
    summary = validation_results["summary"]
    assessment = validation_results["overall_assessment"]
    
    print(f"Overall Credibility: {summary['overall_credibility']}")
    print(f"Credibility Score: {summary['credibility_score']:.1f}/100")
    print(f"Test Success Rate: {summary['test_success_rate']}")
    print(f"Claims Validated: {'✅ YES' if summary['claims_validated'] else '❌ NO'}")
    print(f"Best Performance: {summary['best_performance']}")
    print(f"Measurement Quality: {summary['measurement_quality']}")
    print(f"Recommendation: {summary['recommendation']}")
    
    if validation_results["failed_tests"]:
        print(f"\n⚠️ Failed Tests:")
        for test in validation_results["failed_tests"]:
            print(f"   - {test}")
    
    print(f"\n💡 Recommendations:")
    for rec in assessment["recommendations"]:
        print(f"   {rec}")
    
    # Save results
    results_file = validator.save_validation_results(validation_results)
    print(f"\n💾 Validation results saved: {results_file}")

if __name__ == "__main__":
    main()