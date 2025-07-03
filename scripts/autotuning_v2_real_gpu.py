#!/usr/bin/env python3
"""
GPU Autotuning V2 - Real Hardware Performance Testing
Uses actual fixed Mojo kernels with production-scale data on Lambda Cloud A10 GPUs
"""

import asyncio
import json
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

@dataclass
class AutotuningV2Config:
    """Configuration for autotuning v2 session."""
    tile_size: int
    block_size: int
    shared_memory_kb: int
    corpus_size: int
    vector_dims: int
    test_queries: int = 100
    iterations: int = 5

@dataclass  
class RealPerformanceMetrics:
    """Real GPU performance measurements."""
    latency_ms: float
    throughput_vectors_per_sec: float
    memory_bandwidth_gb_per_sec: float
    gpu_occupancy_percent: float
    success_rate: float
    error_count: int
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class AutotuningV2Session:
    """Manages autotuning v2 session with real GPU testing."""
    session_id: str
    start_time: datetime
    gpu_type: str = "Lambda Cloud A10"
    mojo_version: str = "25.4.0"
    kernel_version: str = "v2_fixed"

class AutotuningV2Manager:
    """Manages real GPU autotuning with fixed Mojo kernels."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "autotuning_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Integration test that will do real benchmarking
        self.benchmark_script = self.project_root / "integration_test_complete.mojo"
        
        # Ensure we have the latest fixed kernels
        self.kernel_dir = self.project_root / "src" / "kernels"
        
    def generate_production_test_matrix(self) -> List[AutotuningV2Config]:
        """Generate comprehensive test matrix for production workloads."""
        configurations = []
        
        # Production-focused parameter ranges
        # Based on A10 GPU specifications and our kernel capabilities
        tile_sizes = [16, 32, 48, 64, 96, 128]
        block_sizes = [32, 64, 128, 256]  
        memory_configs = [4, 8, 16, 32]  # KB
        
        # Production scale corpus sizes to test scalability
        corpus_sizes = [10_000, 25_000, 50_000]  # Start smaller, scale up
        vector_dims = 768  # Production embedding size
        
        for corpus_size in corpus_sizes:
            for tile in tile_sizes:
                for block in block_sizes:
                    for memory in memory_configs:
                        # Skip configurations that are likely to fail
                        if self.is_valid_configuration(tile, block, memory, corpus_size):
                            configurations.append(AutotuningV2Config(
                                tile_size=tile,
                                block_size=block,
                                shared_memory_kb=memory,
                                corpus_size=corpus_size,
                                vector_dims=vector_dims,
                                test_queries=min(100, corpus_size // 100),
                                iterations=3  # Enough for statistical significance
                            ))
        
        print(f"✅ Generated {len(configurations)} valid test configurations")
        return configurations
    
    def is_valid_configuration(self, tile: int, block: int, memory_kb: int, corpus_size: int) -> bool:
        """Check if configuration is valid for A10 GPU constraints."""
        # A10 GPU constraints
        max_shared_memory_kb = 48  # A10 shared memory per SM
        max_threads_per_block = 1024
        
        # Basic validation
        if memory_kb > max_shared_memory_kb:
            return False
        if tile * block > max_threads_per_block:
            return False
        if corpus_size < 1000:  # Too small to be meaningful
            return False
            
        return True
    
    def create_autotuning_session(self) -> AutotuningV2Session:
        """Create new autotuning v2 session."""
        session_id = f"autotune_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return AutotuningV2Session(
            session_id=session_id,
            start_time=datetime.now()
        )
    
    async def run_real_gpu_benchmark(self, config: AutotuningV2Config) -> Optional[RealPerformanceMetrics]:
        """Run real GPU benchmark using our fixed integration test."""
        print(f"   🔧 Testing: tile={config.tile_size}, block={config.block_size}, "
              f"mem={config.shared_memory_kb}KB, corpus={config.corpus_size:,}")
        
        try:
            # Create enhanced integration test command
            # We'll modify integration_test_complete.mojo to accept these parameters
            cmd = [
                "pixi", "run", "mojo", str(self.benchmark_script),
                "--benchmark-mode",
                f"--tile-size={config.tile_size}",
                f"--block-size={config.block_size}",
                f"--shared-memory-kb={config.shared_memory_kb}",
                f"--corpus-size={config.corpus_size}",
                f"--vector-dims={config.vector_dims}",
                f"--test-queries={config.test_queries}",
                f"--iterations={config.iterations}"
            ]
            
            # Execute real GPU benchmark
            start_time = time.time()
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await result.communicate()
            end_time = time.time()
            
            if result.returncode != 0:
                print(f"   ❌ Benchmark failed: {stderr.decode()[:200]}...")
                return None
            
            # Parse real performance metrics from output
            metrics = self.parse_benchmark_output(stdout.decode())
            if metrics:
                print(f"   ✅ Latency: {metrics.latency_ms:.2f}ms | "
                      f"Throughput: {metrics.throughput_vectors_per_sec:.1f} vec/sec | "
                      f"GPU: {metrics.gpu_occupancy_percent:.1f}%")
                return metrics
            else:
                print(f"   ⚠️  Failed to parse benchmark output")
                return None
                
        except Exception as e:
            print(f"   💥 Exception during benchmark: {e}")
            return None
    
    def parse_benchmark_output(self, output: str) -> Optional[RealPerformanceMetrics]:
        """Parse real performance metrics from integration test output."""
        try:
            # Look for our performance metrics in the output
            lines = output.split('\n')
            metrics = {}
            
            for line in lines:
                if "Real GPU Latency:" in line:
                    metrics['latency_ms'] = float(line.split(':')[1].strip().replace('ms', ''))
                elif "Throughput:" in line and "vectors/sec" in line:
                    metrics['throughput'] = float(line.split(':')[1].strip().split()[0])
                elif "GPU Occupancy:" in line:
                    metrics['occupancy'] = float(line.split(':')[1].strip().replace('%', ''))
                elif "Memory Bandwidth:" in line:
                    metrics['bandwidth'] = float(line.split(':')[1].strip().split()[0])
                elif "Success Rate:" in line:
                    metrics['success_rate'] = float(line.split(':')[1].strip().replace('%', '')) / 100.0
                elif "Error Count:" in line:
                    metrics['error_count'] = int(line.split(':')[1].strip())
            
            if 'latency_ms' in metrics and 'throughput' in metrics:
                return RealPerformanceMetrics(
                    latency_ms=metrics['latency_ms'],
                    throughput_vectors_per_sec=metrics['throughput'],
                    memory_bandwidth_gb_per_sec=metrics.get('bandwidth', 0.0),
                    gpu_occupancy_percent=metrics.get('occupancy', 0.0),
                    success_rate=metrics.get('success_rate', 1.0),
                    error_count=metrics.get('error_count', 0)
                )
            
            return None
            
        except Exception as e:
            print(f"Error parsing benchmark output: {e}")
            return None
    
    async def run_comprehensive_autotuning(self, session: AutotuningV2Session) -> Dict[str, Any]:
        """Run comprehensive autotuning with real GPU measurements."""
        print("🚀 Autotuning V2 - Real GPU Performance Testing")
        print("=" * 60)
        print(f"📊 Session: {session.session_id}")
        print(f"🔧 GPU: {session.gpu_type}")
        print(f"🐍 Mojo: {session.mojo_version}")
        print(f"⚡ Kernels: {session.kernel_version}")
        print(f"🕐 Started: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate test matrix
        configurations = self.generate_production_test_matrix()
        print(f"🧪 Testing {len(configurations)} configurations")
        
        # Run benchmarks
        results = []
        best_config = None
        best_performance = float('inf')
        successful_tests = 0
        
        for i, config in enumerate(configurations):
            print(f"\n[{i+1:3d}/{len(configurations)}] Configuration:")
            
            # Run real GPU benchmark
            metrics = await self.run_real_gpu_benchmark(config)
            
            result = {
                'config': asdict(config),
                'timestamp': datetime.now().isoformat(),
                'success': metrics is not None
            }
            
            if metrics:
                result['metrics'] = metrics.to_dict()
                successful_tests += 1
                
                # Track best performance
                if metrics.latency_ms < best_performance:
                    best_performance = metrics.latency_ms
                    best_config = result
            else:
                result['error'] = "Benchmark execution failed"
            
            results.append(result)
            
            # Progress update
            progress = ((i + 1) / len(configurations)) * 100
            print(f"   Progress: {progress:.1f}% | "
                  f"Successful: {successful_tests}/{i+1} | "
                  f"Best: {best_performance:.2f}ms")
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['success']]
        latencies = [r['metrics']['latency_ms'] for r in successful_results]
        
        summary = {
            'total_tests': len(results),
            'successful_tests': successful_tests,
            'success_rate': successful_tests / len(results) if results else 0,
            'best_latency_ms': best_performance if best_config else None,
            'avg_latency_ms': np.mean(latencies) if latencies else None,
            'median_latency_ms': np.median(latencies) if latencies else None,
            'std_latency_ms': np.std(latencies) if latencies else None,
            'test_duration_minutes': (datetime.now() - session.start_time).total_seconds() / 60
        }
        
        return {
            'session_info': asdict(session),
            'test_summary': summary,
            'best_configuration': best_config,
            'all_results': results,
            'comparison_with_v1': self.compare_with_v1_results(best_config)
        }
    
    def compare_with_v1_results(self, best_v2_result: Optional[Dict]) -> Dict[str, Any]:
        """Compare V2 results with previous V1 simulated results."""
        try:
            # Load V1 results
            v1_file = self.results_dir / "autotune_20250702_233614_results.json"
            if not v1_file.exists():
                return {"error": "V1 results not found"}
            
            with open(v1_file, 'r') as f:
                v1_data = json.load(f)
            
            if not best_v2_result:
                return {"error": "No successful V2 results to compare"}
            
            v1_latency = v1_data['optimization_results']['best_config']['avg_latency_ms']
            v2_latency = best_v2_result['metrics']['latency_ms']
            
            v1_corpus_size = v1_data['session_info']['corpus_size']
            v2_corpus_size = best_v2_result['config']['corpus_size']
            
            return {
                'v1_simulated_latency_ms': v1_latency,
                'v2_real_latency_ms': v2_latency,
                'reality_check_factor': v2_latency / v1_latency,
                'corpus_scale_factor': v2_corpus_size / v1_corpus_size,
                'conclusion': self.generate_comparison_conclusion(v1_latency, v2_latency, v1_corpus_size, v2_corpus_size)
            }
            
        except Exception as e:
            return {"error": f"Comparison failed: {e}"}
    
    def generate_comparison_conclusion(self, v1_lat: float, v2_lat: float, v1_size: int, v2_size: int) -> str:
        """Generate conclusion about V1 vs V2 results."""
        reality_factor = v2_lat / v1_lat
        scale_factor = v2_size / v1_size
        
        if reality_factor > 5:
            return f"V1 results were SEVERELY optimistic. Real performance is {reality_factor:.1f}x slower with {scale_factor:.1f}x larger corpus."
        elif reality_factor > 2:
            return f"V1 results were optimistic. Real performance is {reality_factor:.1f}x slower with {scale_factor:.1f}x larger corpus."
        elif reality_factor > 1.5:
            return f"V1 results were somewhat optimistic. Real performance is {reality_factor:.1f}x slower with {scale_factor:.1f}x larger corpus."
        else:
            return f"V1 simulation was surprisingly accurate! Real performance is only {reality_factor:.1f}x different."
    
    def save_v2_results(self, session: AutotuningV2Session, results: Dict[str, Any]):
        """Save V2 autotuning results."""
        results_file = self.results_dir / f"{session.session_id}_results.json"
        
        # Add metadata
        final_results = {
            **results,
            'metadata': {
                'autotuning_version': 'v2',
                'testing_approach': 'real_gpu_hardware',
                'gpu_type': session.gpu_type,
                'mojo_version': session.mojo_version,
                'kernel_version': session.kernel_version,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"💾 V2 results saved: {results_file}")
        return results_file
    
    async def run_full_autotuning_v2(self) -> Dict[str, Any]:
        """Run complete V2 autotuning session."""
        # Create session
        session = self.create_autotuning_session()
        
        # Run comprehensive testing
        results = await self.run_comprehensive_autotuning(session)
        
        # Save results
        results_file = self.save_v2_results(session, results)
        
        # Print summary
        self.print_final_summary(results)
        
        return {
            'session': session,
            'results': results,
            'results_file': results_file
        }
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Print comprehensive summary of V2 autotuning."""
        summary = results['test_summary']
        best = results['best_configuration']
        comparison = results['comparison_with_v1']
        
        print(f"\n🎉 Autotuning V2 Complete!")
        print("=" * 60)
        print(f"⏱️  Duration: {summary['test_duration_minutes']:.1f} minutes")
        print(f"🧪 Total tests: {summary['total_tests']}")
        print(f"✅ Successful: {summary['successful_tests']} ({summary['success_rate']:.1%})")
        
        if best:
            print(f"\n🏆 Best Configuration:")
            config = best['config']
            metrics = best['metrics']
            print(f"   Tile size: {config['tile_size']}")
            print(f"   Block size: {config['block_size']}")
            print(f"   Shared memory: {config['shared_memory_kb']} KB")
            print(f"   Corpus size: {config['corpus_size']:,} vectors")
            print(f"   🚀 Real latency: {metrics['latency_ms']:.2f}ms")
            print(f"   ⚡ Throughput: {metrics['throughput_vectors_per_sec']:.1f} vectors/sec")
            print(f"   💾 GPU occupancy: {metrics['gpu_occupancy_percent']:.1f}%")
        
        if 'conclusion' in comparison:
            print(f"\n📊 V1 vs V2 Comparison:")
            print(f"   {comparison['conclusion']}")
        
        print(f"\n🎯 Production Readiness:")
        if best and best['metrics']['latency_ms'] < 50:
            print("   ✅ READY - Real latency meets production requirements")
        elif best:
            print(f"   ⚠️  NEEDS OPTIMIZATION - {best['metrics']['latency_ms']:.2f}ms may be too slow")
        else:
            print("   ❌ NOT READY - No successful configurations found")

async def main():
    """Main autotuning V2 execution."""
    print("🔥 Mojo GPU Autotuning V2 - Real Hardware Testing")
    print("🎯 Testing fixed kernels with production-scale data")
    print()
    
    manager = AutotuningV2Manager()
    
    # Check prerequisites
    if not manager.benchmark_script.exists():
        print(f"❌ Benchmark script not found: {manager.benchmark_script}")
        print("   Please ensure integration_test_complete.mojo supports benchmark mode")
        return
    
    if not manager.kernel_dir.exists():
        print(f"❌ Kernel directory not found: {manager.kernel_dir}")
        return
    
    print("✅ Prerequisites check passed")
    print("🚀 Starting real GPU autotuning...")
    
    # Run V2 autotuning
    results = await manager.run_full_autotuning_v2()
    
    print(f"\n📋 Results available at: {results['results_file']}")
    print("🎉 Autotuning V2 complete - Real GPU performance data collected!")

if __name__ == "__main__":
    asyncio.run(main())