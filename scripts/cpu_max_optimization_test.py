#!/usr/bin/env python3
"""
CPU MAX Graph Optimization Test
Test MAX Graph optimizations on CPU to validate the approach before GPU testing
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.max_graph.semantic_search_graph import MaxGraphConfig, MaxSemanticSearchGraph

@dataclass
class CPUOptimizationStep:
    """CPU optimization step for testing MAX Graph features."""
    name: str
    description: str
    config: Dict[str, Any]
    expected_benefit: str

class CPUMaxOptimizer:
    """Test MAX Graph optimizations on CPU before GPU deployment."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "data" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Small test parameters for CPU
        self.corpus_size = 5000
        self.vector_dims = 768
        
    def define_cpu_optimization_tests(self) -> List[CPUOptimizationStep]:
        """Define CPU optimization tests."""
        return [
            CPUOptimizationStep(
                name="baseline_cpu",
                description="Baseline MAX Graph on CPU",
                config={
                    "corpus_size": self.corpus_size,
                    "vector_dims": self.vector_dims,
                    "device": "cpu",
                    "use_fp16": False,
                    "enable_fusion": False
                },
                expected_benefit="Establish CPU baseline"
            ),
            CPUOptimizationStep(
                name="fusion_enabled",
                description="Enable automatic kernel fusion",
                config={
                    "corpus_size": self.corpus_size,
                    "vector_dims": self.vector_dims,
                    "device": "cpu",
                    "use_fp16": False,
                    "enable_fusion": True
                },
                expected_benefit="20-30% improvement from kernel fusion"
            ),
            CPUOptimizationStep(
                name="smaller_corpus",
                description="Test with smaller corpus for speed",
                config={
                    "corpus_size": 2000,
                    "vector_dims": self.vector_dims,
                    "device": "cpu",
                    "use_fp16": False,
                    "enable_fusion": True
                },
                expected_benefit="Faster execution, validate scaling"
            )
        ]
    
    def create_test_data(self, corpus_size: int, vector_dims: int):
        """Create consistent test data."""
        np.random.seed(42)
        
        query_embeddings = np.random.randn(3, vector_dims).astype(np.float32)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        corpus_embeddings = np.random.randn(corpus_size, vector_dims).astype(np.float32)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        
        return query_embeddings, corpus_embeddings
    
    def run_cpu_test(self, step: CPUOptimizationStep) -> Optional[Dict[str, Any]]:
        """Run single CPU optimization test."""
        print(f"\n🔧 Testing: {step.name}")
        print(f"   Description: {step.description}")
        print(f"   Expected: {step.expected_benefit}")
        print(f"   Config: {step.config}")
        
        try:
            # Create MAX Graph configuration
            config = MaxGraphConfig(**step.config)
            
            # Initialize MAX Graph
            print("   🏗️  Building MAX Graph...")
            max_search = MaxSemanticSearchGraph(config)
            
            # Compile for CPU
            print("   🚀 Compiling for CPU...")
            max_search.compile("cpu")
            
            # Create test data
            query_embeddings, corpus_embeddings = self.create_test_data(
                config.corpus_size, config.vector_dims
            )
            
            # Warm-up run
            print("   🔥 Warming up...")
            max_search.search_similarity(query_embeddings[0], corpus_embeddings)
            
            # Benchmark runs
            print("   ⏱️  Benchmarking...")
            latencies = []
            for i in range(3):  # Fewer runs for CPU testing
                start_time = time.perf_counter()
                result = max_search.search_similarity(query_embeddings[0], corpus_embeddings)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                print(f"     Run {i+1}: {latency_ms:.1f}ms")
            
            # Calculate metrics
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            throughput = config.corpus_size / (avg_latency / 1000.0)
            
            print(f"   ✅ Average: {avg_latency:.1f}ms")
            print(f"   ⚡ Throughput: {throughput:.0f} vectors/sec")
            
            return {
                'step_name': step.name,
                'config': step.config,
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'throughput_vectors_per_sec': throughput,
                'success': True
            }
            
        except Exception as e:
            print(f"   ❌ Test failed: {str(e)}")
            return {
                'step_name': step.name,
                'config': step.config,
                'error': str(e),
                'success': False
            }
    
    def run_all_cpu_tests(self) -> Dict[str, Any]:
        """Run all CPU optimization tests."""
        print("🚀 CPU MAX Graph Optimization Tests")
        print("=" * 50)
        print(f"🎯 Goal: Validate MAX Graph optimizations on CPU")
        print(f"📊 Corpus: {self.corpus_size:,} vectors")
        print(f"💻 Platform: Local CPU testing")
        print()
        
        # Get test steps
        steps = self.define_cpu_optimization_tests()
        
        # Run tests
        results = []
        baseline_latency = None
        
        for i, step in enumerate(steps):
            print(f"[{i+1}/{len(steps)}] CPU Test:")
            
            result = self.run_cpu_test(step)
            if result:
                results.append(result)
                
                # Track baseline
                if step.name == "baseline_cpu" and result['success']:
                    baseline_latency = result['avg_latency_ms']
                
                # Calculate improvement vs baseline
                if baseline_latency and result['success'] and step.name != "baseline_cpu":
                    improvement = baseline_latency / result['avg_latency_ms']
                    improvement_pct = ((baseline_latency - result['avg_latency_ms']) / baseline_latency) * 100
                    print(f"   📈 vs Baseline: {improvement:.2f}x ({improvement_pct:+.1f}%)")
        
        # Analyze results
        successful_results = [r for r in results if r['success']]
        
        analysis = {
            'total_tests': len(steps),
            'successful_tests': len(successful_results),
            'baseline_latency_ms': baseline_latency,
            'results': results,
            'max_graph_working': len(successful_results) > 0,
            'fusion_effective': False,
            'ready_for_gpu': False
        }
        
        # Check if fusion was effective
        if len(successful_results) >= 2:
            baseline_result = next((r for r in successful_results if r['step_name'] == 'baseline_cpu'), None)
            fusion_result = next((r for r in successful_results if r['step_name'] == 'fusion_enabled'), None)
            
            if baseline_result and fusion_result:
                fusion_improvement = baseline_result['avg_latency_ms'] / fusion_result['avg_latency_ms']
                analysis['fusion_effective'] = fusion_improvement > 1.1  # 10% improvement threshold
                analysis['fusion_improvement'] = fusion_improvement
        
        # Determine if ready for GPU testing
        analysis['ready_for_gpu'] = (
            analysis['max_graph_working'] and 
            analysis['successful_tests'] >= 2
        )
        
        return analysis
    
    def print_cpu_test_summary(self, analysis: Dict[str, Any]):
        """Print CPU test summary."""
        print(f"\n🎉 CPU MAX Graph Test Summary")
        print("=" * 40)
        
        print(f"📊 Results:")
        print(f"   Tests completed: {analysis['successful_tests']}/{analysis['total_tests']}")
        print(f"   MAX Graph working: {'✅ Yes' if analysis['max_graph_working'] else '❌ No'}")
        
        if analysis['baseline_latency_ms']:
            print(f"   CPU baseline: {analysis['baseline_latency_ms']:.1f}ms")
        
        if analysis.get('fusion_effective'):
            improvement = analysis.get('fusion_improvement', 1.0)
            print(f"   Kernel fusion: ✅ Effective ({improvement:.2f}x improvement)")
        elif 'fusion_improvement' in analysis:
            improvement = analysis.get('fusion_improvement', 1.0)
            print(f"   Kernel fusion: ⚠️ Limited benefit ({improvement:.2f}x)")
        
        print(f"\n🎯 Next Steps:")
        if analysis['ready_for_gpu']:
            print("   ✅ Ready for GPU optimization testing")
            print("   💡 Consider Modular Cloud platform for GPU access")
            print("   🚀 GPU optimizations should provide significant speedup")
        else:
            print("   ⚠️ Need to resolve MAX Graph CPU issues first")
            print("   🔧 Debug compilation or environment problems")
        
        print(f"\n💰 Cost Analysis:")
        print(f"   Local CPU testing: FREE")
        print(f"   Modular Platform: FREE (Community Edition)")
        print(f"   Lambda Cloud: $1.50/hour (if needed)")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for result in analysis['results']:
            status = "✅" if result['success'] else "❌"
            name = result['step_name']
            if result['success']:
                latency = result['avg_latency_ms']
                throughput = result['throughput_vectors_per_sec']
                print(f"   {status} {name}: {latency:.1f}ms ({throughput:.0f} vec/sec)")
            else:
                error = result.get('error', 'Unknown error')[:50]
                print(f"   {status} {name}: FAILED - {error}")

def main():
    """Main CPU optimization test."""
    print("🚀 CPU MAX Graph Optimization Test")
    print("🎯 Validate optimizations before GPU deployment")
    print("💻 Testing locally with FREE Modular Community Edition")
    print()
    
    optimizer = CPUMaxOptimizer()
    
    # Run CPU tests
    analysis = optimizer.run_all_cpu_tests()
    
    # Print summary
    optimizer.print_cpu_test_summary(analysis)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = optimizer.results_dir / f"cpu_max_optimization_{timestamp}.json"
    
    # Convert any non-serializable objects to JSON-safe format
    def make_json_serializable(obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
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
    
    serializable_analysis = make_json_serializable(analysis)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")

if __name__ == "__main__":
    main()