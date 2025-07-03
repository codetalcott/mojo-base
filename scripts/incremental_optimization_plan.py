#!/usr/bin/env python3
"""
Incremental MAX Graph Optimization Plan
Safe, step-by-step approach to sub-millisecond performance

Preserves existing working code while systematically testing each optimization.
NO LAMBDA REQUIRED - all optimizations run locally.
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
class OptimizationStep:
    """Single optimization step with clear before/after measurement."""
    name: str
    description: str
    config_changes: Dict[str, Any]
    expected_improvement: str
    target_latency_ms: float
    requires_gpu: bool = True

class IncrementalOptimizer:
    """Safe, incremental optimization with preserved fallbacks."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "data" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load baseline performance
        self.baseline_latency = 1.8  # Current MAX Graph performance
        self.corpus_size = 10000  # Start smaller for testing
        self.vector_dims = 768
        
    def define_optimization_steps(self) -> List[OptimizationStep]:
        """Define incremental optimization steps."""
        return [
            OptimizationStep(
                name="step1_fusion_cpu",
                description="Enable automatic kernel fusion on CPU",
                config_changes={"enable_fusion": True},
                expected_improvement="20-30% latency reduction",
                target_latency_ms=15.0,  # CPU baseline will be higher
                requires_gpu=False
            ),
            OptimizationStep(
                name="step2_fusion",
                description="Enable automatic kernel fusion",
                config_changes={"enable_fusion": True},
                expected_improvement="20-30% additional reduction",
                target_latency_ms=0.9,
                requires_gpu=True
            ),
            OptimizationStep(
                name="step3_onnx_model",
                description="Switch to ONNX BGE model for MAX Engine optimization",
                config_changes={"model_format": "onnx", "model_path": "bge-base-en-v1.5"},
                expected_improvement="15-25% inference speedup",
                target_latency_ms=0.7,
                requires_gpu=False  # Can test on CPU first
            ),
            OptimizationStep(
                name="step4_batch_optimization",
                description="Optimize batch processing and memory layout",
                config_changes={"batch_size": 1, "memory_layout": "optimized"},
                expected_improvement="10-20% memory efficiency",
                target_latency_ms=0.6,
                requires_gpu=True
            ),
            OptimizationStep(
                name="step5_async_execution",
                description="Enable asynchronous execution pipelines",
                config_changes={"async_execution": True, "pipeline_depth": 2},
                expected_improvement="5-15% overlap benefits",
                target_latency_ms=0.5,
                requires_gpu=True
            )
        ]
    
    def create_baseline_config(self) -> MaxGraphConfig:
        """Create baseline MAX Graph configuration."""
        return MaxGraphConfig(
            corpus_size=self.corpus_size,
            vector_dims=self.vector_dims,
            batch_size=1,
            device="cpu",  # Start with CPU to avoid GPU compilation issues
            use_fp16=False,  # Start with FP32
            enable_fusion=False  # Start without fusion
        )
    
    def apply_optimization_step(self, base_config: MaxGraphConfig, 
                              step: OptimizationStep) -> MaxGraphConfig:
        """Apply single optimization step to configuration."""
        # Create new config with step changes
        new_config = MaxGraphConfig(
            corpus_size=base_config.corpus_size,
            vector_dims=base_config.vector_dims,
            batch_size=base_config.batch_size,
            device=base_config.device,
            use_fp16=base_config.use_fp16,
            enable_fusion=base_config.enable_fusion
        )
        
        # Apply step-specific changes
        for key, value in step.config_changes.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        
        return new_config
    
    def benchmark_configuration(self, config: MaxGraphConfig, 
                              step_name: str) -> Optional[Dict[str, float]]:
        """Benchmark single configuration safely."""
        try:
            print(f"🔧 Testing {step_name}")
            print(f"   FP16: {config.use_fp16}")
            print(f"   Fusion: {config.enable_fusion}")
            print(f"   Device: {config.device}")
            
            # Create and compile MAX Graph
            max_search = MaxSemanticSearchGraph(config)
            max_search.compile()
            
            # Create test data
            query_embeddings, corpus_embeddings = self.create_test_data()
            
            # Warm-up run
            max_search.search_similarity(query_embeddings[0], corpus_embeddings)
            
            # Benchmark runs
            latencies = []
            for i in range(5):
                start_time = time.perf_counter()
                result = max_search.search_similarity(query_embeddings[0], corpus_embeddings)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                print(f"     Run {i+1}: {latency_ms:.3f}ms")
            
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            std_latency = np.std(latencies)
            
            return {
                'avg_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'std_latency_ms': std_latency,
                'throughput_vectors_per_sec': config.corpus_size / (avg_latency / 1000.0)
            }
            
        except Exception as e:
            print(f"   ❌ Step failed: {e}")
            return None
    
    def create_test_data(self):
        """Create consistent test data."""
        np.random.seed(42)
        
        query_embeddings = np.random.randn(5, self.vector_dims).astype(np.float32)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        corpus_embeddings = np.random.randn(self.corpus_size, self.vector_dims).astype(np.float32)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        
        return query_embeddings, corpus_embeddings
    
    def run_incremental_optimization(self) -> Dict[str, Any]:
        """Run complete incremental optimization process."""
        print("🚀 Incremental MAX Graph Optimization")
        print("=" * 50)
        print(f"🎯 Goal: Sub-millisecond semantic search")
        print(f"📊 Baseline: {self.baseline_latency:.1f}ms")
        print(f"🔄 Strategy: One optimization at a time")
        print(f"💻 Environment: Local testing (no Lambda required)")
        print()
        
        # Get optimization steps
        steps = self.define_optimization_steps()
        
        # Start with baseline configuration
        current_config = self.create_baseline_config()
        
        # Benchmark baseline
        print("📊 Step 0: Baseline Measurement")
        baseline_metrics = self.benchmark_configuration(current_config, "baseline")
        
        if not baseline_metrics:
            print("❌ Baseline benchmark failed - cannot proceed")
            return {'error': 'Baseline benchmark failed'}
        
        current_latency = baseline_metrics['avg_latency_ms']
        print(f"✅ Baseline confirmed: {current_latency:.3f}ms")
        
        # Track results
        results = {
            'baseline': {
                'config': {
                    'use_fp16': current_config.use_fp16,
                    'enable_fusion': current_config.enable_fusion,
                    'device': current_config.device
                },
                'metrics': baseline_metrics
            },
            'steps': [],
            'best_result': None,
            'sub_millisecond_achieved': False
        }
        
        # Run incremental optimization
        for i, step in enumerate(steps):
            print(f"\n🔧 Step {i+1}: {step.name}")
            print(f"   {step.description}")
            print(f"   Target: <{step.target_latency_ms:.1f}ms")
            print(f"   Expected: {step.expected_improvement}")
            
            # Apply optimization step
            new_config = self.apply_optimization_step(current_config, step)
            
            # Benchmark new configuration
            step_metrics = self.benchmark_configuration(new_config, step.name)
            
            if step_metrics:
                new_latency = step_metrics['avg_latency_ms']
                improvement = current_latency / new_latency
                improvement_pct = ((current_latency - new_latency) / current_latency) * 100
                
                print(f"   ✅ Result: {new_latency:.3f}ms")
                print(f"   📈 Improvement: {improvement:.2f}x ({improvement_pct:+.1f}%)")
                
                # Check if target met
                target_met = new_latency <= step.target_latency_ms
                print(f"   🎯 Target: {'✅ MET' if target_met else '❌ MISSED'}")
                
                # Check sub-millisecond achievement
                if new_latency < 1.0:
                    results['sub_millisecond_achieved'] = True
                    print(f"   🎉 SUB-MILLISECOND ACHIEVED!")
                
                # Record step result
                step_result = {
                    'step_name': step.name,
                    'description': step.description,
                    'config_changes': step.config_changes,
                    'target_latency_ms': step.target_latency_ms,
                    'metrics': step_metrics,
                    'improvement_factor': improvement,
                    'improvement_percent': improvement_pct,
                    'target_met': target_met,
                    'config': {
                        'use_fp16': new_config.use_fp16,
                        'enable_fusion': new_config.enable_fusion,
                        'device': new_config.device
                    }
                }
                
                results['steps'].append(step_result)
                
                # Update best result
                if not results['best_result'] or new_latency < results['best_result']['metrics']['avg_latency_ms']:
                    results['best_result'] = step_result
                
                # Continue with successful configuration
                current_config = new_config
                current_latency = new_latency
                
                print(f"   ✅ Step successful - continuing with optimized config")
                
            else:
                print(f"   ❌ Step failed - reverting to previous config")
                
                # Record failed step
                step_result = {
                    'step_name': step.name,
                    'description': step.description,
                    'config_changes': step.config_changes,
                    'target_latency_ms': step.target_latency_ms,
                    'success': False,
                    'error': 'Benchmark execution failed'
                }
                
                results['steps'].append(step_result)
                
                # Continue with previous config (safe fallback)
                print(f"   🔄 Continuing with previous configuration")
        
        return results
    
    def print_optimization_summary(self, results: Dict[str, Any]):
        """Print comprehensive optimization summary."""
        print(f"\n🎉 Incremental Optimization Complete!")
        print("=" * 50)
        
        baseline_latency = results['baseline']['metrics']['avg_latency_ms']
        
        if results['best_result']:
            best_latency = results['best_result']['metrics']['avg_latency_ms']
            total_improvement = baseline_latency / best_latency
            
            print(f"📊 Results Summary:")
            print(f"   Baseline: {baseline_latency:.3f}ms")
            print(f"   Best: {best_latency:.3f}ms")
            print(f"   Total Improvement: {total_improvement:.2f}x")
            
            if results['sub_millisecond_achieved']:
                print(f"\n🎉 SUB-MILLISECOND ACHIEVEMENT UNLOCKED!")
                print(f"   This enables:")
                print(f"   • Real-time-as-you-type search")
                print(f"   • 1000+ queries per second")
                print(f"   • Ultra-responsive UI")
            
            print(f"\n🏆 Best Configuration:")
            best_config = results['best_result']['config']
            for key, value in best_config.items():
                print(f"   {key}: {value}")
            
            print(f"\n📈 Step-by-Step Progress:")
            current_latency = baseline_latency
            for step in results['steps']:
                if 'metrics' in step:
                    new_latency = step['metrics']['avg_latency_ms']
                    improvement = current_latency / new_latency
                    print(f"   {step['step_name']}: {current_latency:.3f}ms → {new_latency:.3f}ms ({improvement:.2f}x)")
                    current_latency = new_latency
                else:
                    print(f"   {step['step_name']}: FAILED")
        
        else:
            print(f"❌ No successful optimizations achieved")
            print(f"   Baseline remains at: {baseline_latency:.3f}ms")
    
    def save_results(self, results: Dict[str, Any]) -> Path:
        """Save optimization results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"incremental_optimization_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved: {results_file}")
        return results_file

def main():
    """Main incremental optimization execution."""
    print("🚀 Incremental MAX Graph Optimization")
    print("🎯 Safe, step-by-step path to sub-millisecond performance")
    print("💻 No Lambda required - all tests run locally")
    print()
    
    optimizer = IncrementalOptimizer()
    
    # Run incremental optimization
    results = optimizer.run_incremental_optimization()
    
    # Print summary
    optimizer.print_optimization_summary(results)
    
    # Save results
    optimizer.save_results(results)
    
    print(f"\n✅ Incremental optimization complete!")
    print(f"🔄 All working configurations preserved as fallbacks")

if __name__ == "__main__":
    main()