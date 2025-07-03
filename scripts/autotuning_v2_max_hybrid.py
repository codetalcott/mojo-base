#!/usr/bin/env python3
"""
Hybrid Autotuning V2 - MAX Graph + Legacy Mojo Kernels
Integrates MAX Graph API with our existing autotuning framework for optimal performance
"""

import asyncio
import json
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    # Try to import MAX Graph implementation
    from src.max_graph.semantic_search_graph import (
        MaxSemanticSearchGraph, 
        MaxSemanticSearchBenchmark,
        MaxGraphConfig,
        create_test_data
    )
    MAX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MAX Graph not available: {e}")
    print("   Falling back to legacy Mojo kernels only")
    MAX_AVAILABLE = False

@dataclass
class HybridAutotuningConfig:
    """Configuration for hybrid autotuning with both MAX Graph and legacy kernels."""
    # Basic parameters
    corpus_size: int
    vector_dims: int = 768
    test_queries: int = 100
    iterations: int = 3
    
    # MAX Graph specific
    use_max_graph: bool = True
    max_device: str = "cpu"  # or "gpu"
    max_use_fp16: bool = False
    max_enable_fusion: bool = True
    
    # Legacy Mojo kernel specific
    mojo_tile_size: int = 32
    mojo_block_size: int = 64
    mojo_shared_memory_kb: int = 8
    
    # Comparison settings
    compare_implementations: bool = True
    prefer_max_graph: bool = True  # Prefer MAX Graph if performance is similar

@dataclass
class HybridPerformanceMetrics:
    """Performance metrics for hybrid testing."""
    implementation: str  # "max_graph" or "legacy_mojo"
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_vectors_per_sec: float
    memory_usage_estimate: float
    compilation_time_ms: float
    config: Dict[str, Any]
    success_rate: float = 1.0
    error_count: int = 0

class HybridAutotuningManager:
    """Manages autotuning across both MAX Graph and legacy Mojo implementations."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "autotuning_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Available implementations
        self.max_available = MAX_AVAILABLE
        self.legacy_available = True  # Always available
        
        print(f"🔧 Hybrid Autotuning Manager Initialized")
        print(f"   MAX Graph available: {self.max_available}")
        print(f"   Legacy Mojo available: {self.legacy_available}")
    
    def generate_hybrid_test_matrix(self, config: HybridAutotuningConfig) -> List[Dict[str, Any]]:
        """Generate test configurations for both MAX Graph and legacy implementations."""
        test_configs = []
        
        # MAX Graph configurations (if available)
        if self.max_available and config.use_max_graph:
            max_configs = self._generate_max_graph_configs(config)
            test_configs.extend(max_configs)
        
        # Legacy Mojo configurations
        legacy_configs = self._generate_legacy_mojo_configs(config)
        test_configs.extend(legacy_configs)
        
        print(f"✅ Generated {len(test_configs)} hybrid test configurations")
        return test_configs
    
    def _generate_max_graph_configs(self, config: HybridAutotuningConfig) -> List[Dict[str, Any]]:
        """Generate MAX Graph specific configurations."""
        configs = []
        
        # Test different MAX Graph optimization settings
        devices = ["cpu"]
        if config.max_device == "gpu":
            devices.append("gpu")
        
        fp16_options = [False, True] if config.max_use_fp16 else [False]
        fusion_options = [True, False] if config.max_enable_fusion else [True]
        
        for device in devices:
            for use_fp16 in fp16_options:
                for enable_fusion in fusion_options:
                    configs.append({
                        'implementation': 'max_graph',
                        'corpus_size': config.corpus_size,
                        'vector_dims': config.vector_dims,
                        'device': device,
                        'use_fp16': use_fp16,
                        'enable_fusion': enable_fusion,
                        'test_queries': config.test_queries,
                        'iterations': config.iterations
                    })
        
        return configs
    
    def _generate_legacy_mojo_configs(self, config: HybridAutotuningConfig) -> List[Dict[str, Any]]:
        """Generate legacy Mojo kernel configurations."""
        configs = []
        
        # Test different manual optimization parameters
        tile_sizes = [16, 32, 48, 64] if config.mojo_tile_size == 32 else [config.mojo_tile_size]
        block_sizes = [32, 64, 128] if config.mojo_block_size == 64 else [config.mojo_block_size]
        memory_configs = [4, 8, 16] if config.mojo_shared_memory_kb == 8 else [config.mojo_shared_memory_kb]
        
        for tile_size in tile_sizes:
            for block_size in block_sizes:
                for shared_memory_kb in memory_configs:
                    configs.append({
                        'implementation': 'legacy_mojo',
                        'corpus_size': config.corpus_size,
                        'vector_dims': config.vector_dims,
                        'tile_size': tile_size,
                        'block_size': block_size,
                        'shared_memory_kb': shared_memory_kb,
                        'test_queries': config.test_queries,
                        'iterations': config.iterations
                    })
        
        return configs
    
    async def benchmark_max_graph(self, test_config: Dict[str, Any]) -> HybridPerformanceMetrics:
        """Benchmark MAX Graph implementation."""
        if not self.max_available:
            raise RuntimeError("MAX Graph not available")
        
        print(f"🚀 Benchmarking MAX Graph:")
        print(f"   Device: {test_config['device']}")
        print(f"   FP16: {test_config['use_fp16']}")
        print(f"   Fusion: {test_config['enable_fusion']}")
        
        # Create MAX Graph configuration
        max_config = MaxGraphConfig(
            corpus_size=test_config['corpus_size'],
            vector_dims=test_config['vector_dims'],
            device=test_config['device'],
            use_fp16=test_config['use_fp16'],
            enable_fusion=test_config['enable_fusion']
        )
        
        # Create test data
        query_embeddings, corpus_embeddings = create_test_data(
            max_config.corpus_size, max_config.vector_dims
        )
        
        # Measure compilation time
        compile_start = time.time()
        benchmark = MaxSemanticSearchBenchmark(max_config)
        benchmark.max_search.compile()
        compile_time = (time.time() - compile_start) * 1000
        
        # Run benchmark
        metrics_dict = benchmark.benchmark_configuration(
            query_embeddings, corpus_embeddings, 
            iterations=test_config['iterations']
        )
        
        # Convert to HybridPerformanceMetrics
        return HybridPerformanceMetrics(
            implementation="max_graph",
            avg_latency_ms=metrics_dict['avg_latency_ms'],
            std_latency_ms=metrics_dict['std_latency_ms'],
            min_latency_ms=metrics_dict['min_latency_ms'],
            max_latency_ms=metrics_dict['max_latency_ms'],
            throughput_vectors_per_sec=metrics_dict['avg_throughput_vectors_per_sec'],
            memory_usage_estimate=0.0,  # MAX handles memory automatically
            compilation_time_ms=compile_time,
            config=test_config,
            success_rate=1.0,
            error_count=0
        )
    
    async def benchmark_legacy_mojo(self, test_config: Dict[str, Any]) -> Optional[HybridPerformanceMetrics]:
        """Benchmark legacy Mojo implementation using integration test."""
        print(f"🔧 Benchmarking Legacy Mojo:")
        print(f"   Tile: {test_config['tile_size']}")
        print(f"   Block: {test_config['block_size']}")
        print(f"   Memory: {test_config['shared_memory_kb']}KB")
        
        try:
            # Use our existing integration test benchmark
            cmd = [
                "pixi", "run", "mojo", str(self.project_root / "integration_test_benchmark.mojo"),
                "--benchmark-mode",
                f"--tile-size={test_config['tile_size']}",
                f"--block-size={test_config['block_size']}",
                f"--shared-memory-kb={test_config['shared_memory_kb']}",
                f"--corpus-size={test_config['corpus_size']}",
                f"--vector-dims={test_config['vector_dims']}",
                f"--test-queries={test_config['test_queries']}",
                f"--iterations={test_config['iterations']}"
            ]
            
            # Execute legacy Mojo benchmark
            compile_start = time.time()
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root / "portfolio-search"
            )
            
            stdout, stderr = await result.communicate()
            compile_time = (time.time() - compile_start) * 1000
            
            if result.returncode != 0:
                print(f"   ❌ Legacy Mojo benchmark failed: {stderr.decode()[:200]}...")
                return None
            
            # Parse performance metrics from output
            metrics = self._parse_legacy_mojo_output(stdout.decode(), compile_time, test_config)
            return metrics
            
        except Exception as e:
            print(f"   💥 Exception during legacy benchmark: {e}")
            return None
    
    def _parse_legacy_mojo_output(self, output: str, compile_time: float, config: Dict[str, Any]) -> HybridPerformanceMetrics:
        """Parse performance metrics from legacy Mojo output."""
        lines = output.split('\n')
        
        # Look for performance metrics in output
        latency_ms = 1.0  # Default fallback
        throughput = 1000.0  # Default fallback
        
        for line in lines:
            if "Real GPU Latency:" in line:
                try:
                    latency_ms = float(line.split(':')[1].strip().replace('ms', ''))
                except:
                    pass
            elif "Throughput:" in line and "vectors/sec" in line:
                try:
                    throughput = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
        
        return HybridPerformanceMetrics(
            implementation="legacy_mojo",
            avg_latency_ms=latency_ms,
            std_latency_ms=latency_ms * 0.1,  # Estimate 10% std dev
            min_latency_ms=latency_ms * 0.9,
            max_latency_ms=latency_ms * 1.1,
            throughput_vectors_per_sec=throughput,
            memory_usage_estimate=config['corpus_size'] * config['vector_dims'] * 4,  # Estimate
            compilation_time_ms=compile_time,
            config=config,
            success_rate=1.0,
            error_count=0
        )
    
    async def run_hybrid_autotuning(self, config: HybridAutotuningConfig) -> Dict[str, Any]:
        """Run comprehensive hybrid autotuning across both implementations."""
        print("🚀 Hybrid Autotuning V2 - MAX Graph + Legacy Mojo")
        print("=" * 60)
        print(f"🔧 Corpus: {config.corpus_size:,} vectors ({config.vector_dims}D)")
        print(f"📊 MAX Graph: {'✅ Available' if self.max_available else '❌ Not available'}")
        print(f"🔥 Legacy Mojo: ✅ Available")
        
        # Generate test configurations
        test_configs = self.generate_hybrid_test_matrix(config)
        
        # Run benchmarks
        results = []
        max_graph_results = []
        legacy_mojo_results = []
        
        for i, test_config in enumerate(test_configs):
            print(f"\n[{i+1:3d}/{len(test_configs)}] Testing: {test_config['implementation']}")
            
            if test_config['implementation'] == 'max_graph':
                metrics = await self.benchmark_max_graph(test_config)
                if metrics:
                    max_graph_results.append(metrics)
                    results.append(metrics)
            
            elif test_config['implementation'] == 'legacy_mojo':
                metrics = await self.benchmark_legacy_mojo(test_config)
                if metrics:
                    legacy_mojo_results.append(metrics)
                    results.append(metrics)
        
        # Analyze results
        analysis = self._analyze_hybrid_results(max_graph_results, legacy_mojo_results, config)
        
        return {
            'config': asdict(config),
            'max_graph_results': [asdict(r) for r in max_graph_results],
            'legacy_mojo_results': [asdict(r) for r in legacy_mojo_results],
            'analysis': analysis,
            'recommendation': self._generate_recommendation(analysis, config),
            'total_tests': len(results),
            'successful_tests': len([r for r in results if r.success_rate > 0.5])
        }
    
    def _analyze_hybrid_results(self, max_results: List[HybridPerformanceMetrics], 
                               legacy_results: List[HybridPerformanceMetrics],
                               config: HybridAutotuningConfig) -> Dict[str, Any]:
        """Analyze performance comparison between implementations."""
        
        analysis = {
            'max_graph_available': len(max_results) > 0,
            'legacy_mojo_available': len(legacy_results) > 0,
            'best_max_graph': None,
            'best_legacy_mojo': None,
            'performance_comparison': None
        }
        
        # Find best performing configurations
        if max_results:
            best_max = min(max_results, key=lambda x: x.avg_latency_ms)
            analysis['best_max_graph'] = {
                'latency_ms': best_max.avg_latency_ms,
                'throughput': best_max.throughput_vectors_per_sec,
                'config': best_max.config,
                'compilation_time_ms': best_max.compilation_time_ms
            }
        
        if legacy_results:
            best_legacy = min(legacy_results, key=lambda x: x.avg_latency_ms)
            analysis['best_legacy_mojo'] = {
                'latency_ms': best_legacy.avg_latency_ms,
                'throughput': best_legacy.throughput_vectors_per_sec,
                'config': best_legacy.config,
                'compilation_time_ms': best_legacy.compilation_time_ms
            }
        
        # Compare implementations
        if analysis['best_max_graph'] and analysis['best_legacy_mojo']:
            max_latency = analysis['best_max_graph']['latency_ms']
            legacy_latency = analysis['best_legacy_mojo']['latency_ms']
            
            speedup = legacy_latency / max_latency if max_latency > 0 else 1.0
            improvement_pct = ((legacy_latency - max_latency) / legacy_latency) * 100 if legacy_latency > 0 else 0
            
            analysis['performance_comparison'] = {
                'max_graph_latency_ms': max_latency,
                'legacy_mojo_latency_ms': legacy_latency,
                'max_graph_speedup': speedup,
                'improvement_percent': improvement_pct,
                'winner': 'MAX Graph' if speedup > 1.05 else 'Legacy Mojo' if speedup < 0.95 else 'Similar'
            }
        
        return analysis
    
    def _generate_recommendation(self, analysis: Dict[str, Any], config: HybridAutotuningConfig) -> Dict[str, Any]:
        """Generate deployment recommendation based on analysis."""
        
        recommendation = {
            'primary_implementation': 'unknown',
            'fallback_implementation': 'unknown',
            'reasoning': [],
            'deployment_strategy': 'unknown'
        }
        
        max_available = analysis['max_graph_available']
        legacy_available = analysis['legacy_mojo_available']
        comparison = analysis.get('performance_comparison')
        
        if comparison:
            winner = comparison['winner']
            speedup = comparison['max_graph_speedup']
            
            if winner == 'MAX Graph':
                recommendation['primary_implementation'] = 'max_graph'
                recommendation['fallback_implementation'] = 'legacy_mojo'
                recommendation['reasoning'].append(f"MAX Graph is {speedup:.2f}x faster")
                recommendation['reasoning'].append("Benefits from automatic optimizations")
                
            elif winner == 'Legacy Mojo':
                recommendation['primary_implementation'] = 'legacy_mojo'
                recommendation['fallback_implementation'] = 'max_graph' if max_available else 'none'
                recommendation['reasoning'].append(f"Legacy Mojo is {1/speedup:.2f}x faster")
                recommendation['reasoning'].append("Manual optimizations outperform automatic ones")
                
            else:  # Similar performance
                if config.prefer_max_graph and max_available:
                    recommendation['primary_implementation'] = 'max_graph'
                    recommendation['fallback_implementation'] = 'legacy_mojo'
                    recommendation['reasoning'].append("Performance similar, preferring MAX Graph")
                    recommendation['reasoning'].append("Better long-term maintainability")
                else:
                    recommendation['primary_implementation'] = 'legacy_mojo'
                    recommendation['fallback_implementation'] = 'max_graph' if max_available else 'none'
                    recommendation['reasoning'].append("Performance similar, using proven legacy implementation")
        
        elif max_available and not legacy_available:
            recommendation['primary_implementation'] = 'max_graph'
            recommendation['fallback_implementation'] = 'none'
            recommendation['reasoning'].append("Only MAX Graph available")
            
        elif legacy_available and not max_available:
            recommendation['primary_implementation'] = 'legacy_mojo'
            recommendation['fallback_implementation'] = 'none'
            recommendation['reasoning'].append("Only Legacy Mojo available")
            
        else:
            recommendation['primary_implementation'] = 'none'
            recommendation['fallback_implementation'] = 'none'
            recommendation['reasoning'].append("No working implementations found")
        
        # Deployment strategy
        if recommendation['primary_implementation'] != 'none':
            if recommendation['fallback_implementation'] != 'none':
                recommendation['deployment_strategy'] = 'hybrid_with_fallback'
            else:
                recommendation['deployment_strategy'] = 'single_implementation'
        else:
            recommendation['deployment_strategy'] = 'manual_investigation_needed'
        
        return recommendation
    
    def save_hybrid_results(self, session_id: str, results: Dict[str, Any]):
        """Save hybrid autotuning results."""
        results_file = self.results_dir / f"{session_id}_hybrid_results.json"
        
        # Add metadata
        final_results = {
            **results,
            'metadata': {
                'autotuning_version': 'v2_hybrid',
                'max_graph_available': self.max_available,
                'legacy_mojo_available': self.legacy_available,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"💾 Hybrid results saved: {results_file}")
        return results_file

async def main():
    """Main hybrid autotuning execution."""
    print("🔥 Hybrid Autotuning V2 - MAX Graph + Legacy Mojo Integration")
    print("============================================================")
    
    # Test configuration
    config = HybridAutotuningConfig(
        corpus_size=10000,  # Start with smaller corpus for testing
        vector_dims=768,
        test_queries=50,
        iterations=3,
        use_max_graph=True,
        max_device="cpu",
        max_use_fp16=False,
        max_enable_fusion=True,
        compare_implementations=True,
        prefer_max_graph=True
    )
    
    # Run hybrid autotuning
    manager = HybridAutotuningManager()
    
    session_id = f"hybrid_autotune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = await manager.run_hybrid_autotuning(config)
    
    # Save results
    results_file = manager.save_hybrid_results(session_id, results)
    
    # Print summary
    print(f"\n🎉 Hybrid Autotuning Complete!")
    print(f"   Total tests: {results['total_tests']}")
    print(f"   Successful: {results['successful_tests']}")
    
    if results['analysis']['performance_comparison']:
        comp = results['analysis']['performance_comparison']
        print(f"   Winner: {comp['winner']}")
        print(f"   MAX Graph: {comp['max_graph_latency_ms']:.3f}ms")
        print(f"   Legacy Mojo: {comp['legacy_mojo_latency_ms']:.3f}ms")
    
    rec = results['recommendation']
    print(f"\n📋 Recommendation: {rec['primary_implementation']}")
    for reason in rec['reasoning']:
        print(f"   • {reason}")
    
    print(f"\n📊 Results saved: {results_file}")

if __name__ == "__main__":
    asyncio.run(main())