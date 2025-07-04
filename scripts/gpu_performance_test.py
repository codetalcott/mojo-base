#!/usr/bin/env python3
"""
GPU Performance Testing Script for Modular Platform
Tests MAX Graph performance on GPU hardware with comprehensive benchmarks.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from max_graph.semantic_search_graph import MaxGraphConfig, MaxSemanticSearchGraph, create_test_data
    MAX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MAX Graph not available: {e}")
    MAX_AVAILABLE = False

def detect_gpu_capabilities():
    """Detect available GPU hardware and capabilities."""
    print("🔍 Detecting GPU Capabilities")
    print("=" * 40)
    
    gpu_info = {
        'cuda_available': False,
        'gpu_devices': [],
        'recommended_config': None
    }
    
    try:
        # Try to detect CUDA
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info['cuda_available'] = True
            gpu_lines = result.stdout.strip().split('\n')
            for line in gpu_lines:
                if line.strip():
                    name, memory = line.split(',')
                    gpu_info['gpu_devices'].append({
                        'name': name.strip(),
                        'memory_mb': int(memory.strip().split()[0])
                    })
    except Exception as e:
        print(f"   CUDA detection failed: {e}")
    
    # Check for other GPU types
    try:
        # Apple Metal detection (basic)
        import platform
        if platform.system() == 'Darwin':
            gpu_info['metal_available'] = True
            gpu_info['gpu_devices'].append({
                'name': 'Apple Metal GPU',
                'memory_mb': 'Unknown'
            })
    except Exception:
        pass
    
    if gpu_info['gpu_devices']:
        print(f"✅ Found {len(gpu_info['gpu_devices'])} GPU(s):")
        for i, gpu in enumerate(gpu_info['gpu_devices']):
            print(f"   {i+1}. {gpu['name']} ({gpu['memory_mb']} MB)")
        
        # Recommend configuration based on memory
        if gpu_info['cuda_available']:
            gpu_info['recommended_config'] = {
                'device': 'cuda',
                'use_fp16': True,
                'enable_fusion': True,
                'max_corpus_size': 50000  # Conservative estimate
            }
        else:
            gpu_info['recommended_config'] = {
                'device': 'gpu',
                'use_fp16': False,
                'enable_fusion': True,
                'max_corpus_size': 25000
            }
    else:
        print("❌ No GPU devices detected")
        gpu_info['recommended_config'] = {
            'device': 'cpu',
            'use_fp16': False,
            'enable_fusion': False,
            'max_corpus_size': 10000
        }
    
    return gpu_info

def test_gpu_compilation():
    """Test if MAX Graph can compile for GPU."""
    if not MAX_AVAILABLE:
        print("❌ MAX Graph not available for testing")
        return False
    
    print("\n🔧 Testing GPU Compilation")
    print("=" * 35)
    
    try:
        # Test small configuration first
        config = MaxGraphConfig(
            corpus_size=1000,
            vector_dims=768,
            device="gpu",
            use_fp16=False,
            enable_fusion=True
        )
        
        print(f"   Config: {config.corpus_size} vectors, device={config.device}")
        print(f"   FP16: {config.use_fp16}, Fusion: {config.enable_fusion}")
        
        # Try to create and compile
        max_search = MaxSemanticSearchGraph(config)
        max_search.compile()
        
        if max_search.model is not None:
            print("✅ GPU compilation successful!")
            return True
        else:
            print("❌ GPU compilation failed - model is None")
            return False
            
    except Exception as e:
        print(f"❌ GPU compilation failed: {e}")
        return False

def benchmark_gpu_performance(corpus_sizes: List[int], iterations: int = 5):
    """Benchmark GPU performance across different corpus sizes."""
    if not MAX_AVAILABLE:
        print("❌ MAX Graph not available for GPU benchmarking")
        return {}
    
    print(f"\n🚀 GPU Performance Benchmark")
    print("=" * 40)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_results': [],
        'summary': {}
    }
    
    for corpus_size in corpus_sizes:
        print(f"\n📊 Testing {corpus_size:,} vectors...")
        
        # Test configurations
        configs_to_test = [
            {
                'name': f'gpu_fp32_{corpus_size}',
                'config': MaxGraphConfig(
                    corpus_size=corpus_size,
                    device="gpu",
                    use_fp16=False,
                    enable_fusion=True
                )
            },
            {
                'name': f'gpu_fp16_{corpus_size}',
                'config': MaxGraphConfig(
                    corpus_size=corpus_size,
                    device="gpu",
                    use_fp16=True,
                    enable_fusion=True
                )
            }
        ]
        
        for test_config in configs_to_test:
            try:
                config = test_config['config']
                print(f"   {test_config['name']}: ", end="")
                
                # Create test data
                query_embeddings, corpus_embeddings = create_test_data(
                    corpus_size, config.vector_dims
                )
                
                # Create and compile graph
                max_search = MaxSemanticSearchGraph(config)
                max_search.compile()
                
                if max_search.model is None:
                    print("❌ Compilation failed")
                    continue
                
                # Benchmark
                latencies = []
                for i in range(iterations):
                    result = max_search.search_similarity(
                        query_embeddings[0], corpus_embeddings
                    )
                    latencies.append(result['execution_time_ms'])
                
                avg_latency = np.mean(latencies)
                min_latency = np.min(latencies)
                throughput = corpus_size / (avg_latency / 1000.0)
                
                test_result = {
                    'test_name': test_config['name'],
                    'corpus_size': corpus_size,
                    'device': config.device,
                    'use_fp16': config.use_fp16,
                    'enable_fusion': config.enable_fusion,
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min_latency,
                    'throughput_vectors_per_sec': throughput,
                    'latencies': latencies,
                    'success': True
                }
                
                results['test_results'].append(test_result)
                
                print(f"✅ {avg_latency:.3f}ms avg ({min_latency:.3f}ms min)")
                
            except Exception as e:
                print(f"❌ {str(e)}")
                test_result = {
                    'test_name': test_config['name'],
                    'corpus_size': corpus_size,
                    'error': str(e),
                    'success': False
                }
                results['test_results'].append(test_result)
    
    return results

def analyze_gpu_results(results: Dict[str, Any]):
    """Analyze GPU benchmark results."""
    if not results.get('test_results'):
        print("❌ No GPU results to analyze")
        return
    
    print(f"\n📈 GPU Performance Analysis")
    print("=" * 40)
    
    successful_tests = [r for r in results['test_results'] if r.get('success')]
    
    if not successful_tests:
        print("❌ No successful GPU tests to analyze")
        return
    
    print(f"✅ {len(successful_tests)} successful tests")
    
    # Find best performance
    best_test = min(successful_tests, key=lambda x: x['avg_latency_ms'])
    print(f"\n🏆 Best Performance:")
    print(f"   Test: {best_test['test_name']}")
    print(f"   Latency: {best_test['avg_latency_ms']:.3f}ms")
    print(f"   Throughput: {best_test['throughput_vectors_per_sec']:.0f} vectors/sec")
    
    # Compare FP16 vs FP32
    fp16_tests = [r for r in successful_tests if r.get('use_fp16')]
    fp32_tests = [r for r in successful_tests if not r.get('use_fp16')]
    
    if fp16_tests and fp32_tests:
        print(f"\n⚡ FP16 vs FP32 Comparison:")
        for corpus_size in set(r['corpus_size'] for r in successful_tests):
            fp16_result = next((r for r in fp16_tests if r['corpus_size'] == corpus_size), None)
            fp32_result = next((r for r in fp32_tests if r['corpus_size'] == corpus_size), None)
            
            if fp16_result and fp32_result:
                speedup = fp32_result['avg_latency_ms'] / fp16_result['avg_latency_ms']
                print(f"   {corpus_size:,} vectors: {speedup:.2f}x speedup with FP16")
    
    # Calculate scaling efficiency
    print(f"\n📊 Scaling Analysis:")
    corpus_sizes = sorted(set(r['corpus_size'] for r in successful_tests))
    for i, size in enumerate(corpus_sizes):
        size_tests = [r for r in successful_tests if r['corpus_size'] == size]
        if size_tests:
            avg_latency = np.mean([r['avg_latency_ms'] for r in size_tests])
            per_1k_ms = avg_latency / (size / 1000)
            print(f"   {size:,} vectors: {avg_latency:.3f}ms ({per_1k_ms:.3f}ms per 1K)")

def compare_gpu_vs_cpu(gpu_results: Dict[str, Any]):
    """Compare GPU results with our CPU baseline."""
    print(f"\n🔄 GPU vs CPU Comparison")
    print("=" * 35)
    
    # Our CPU baseline (from previous tests)
    cpu_baseline = {
        2000: 0.910,  # ms
        5000: 1.805,  # ms
        10000: 3.682  # ms
    }
    
    successful_gpu = [r for r in gpu_results.get('test_results', []) if r.get('success')]
    
    if not successful_gpu:
        print("❌ No GPU results for comparison")
        return
    
    print("Performance Comparison:")
    print("Size     | CPU (ms) | GPU (ms) | Speedup")
    print("-" * 45)
    
    for corpus_size in sorted(cpu_baseline.keys()):
        cpu_time = cpu_baseline[corpus_size]
        
        # Find best GPU result for this size
        gpu_tests = [r for r in successful_gpu if r['corpus_size'] == corpus_size]
        if gpu_tests:
            best_gpu = min(gpu_tests, key=lambda x: x['avg_latency_ms'])
            gpu_time = best_gpu['avg_latency_ms']
            speedup = cpu_time / gpu_time
            
            print(f"{corpus_size:,} | {cpu_time:7.3f} | {gpu_time:7.3f} | {speedup:6.2f}x")
        else:
            print(f"{corpus_size:,} | {cpu_time:7.3f} | No data  | N/A")

def save_gpu_results(results: Dict[str, Any], filename: str = None):
    """Save GPU benchmark results to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/results/gpu_performance_{timestamp}.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def make_json_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    serializable_results = make_json_serializable(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Results saved to {filename}")

def main():
    """Main GPU testing workflow."""
    print("🚀 Modular Platform GPU Performance Testing")
    print("=" * 60)
    
    # Step 1: Detect GPU capabilities
    gpu_info = detect_gpu_capabilities()
    
    if not gpu_info['gpu_devices']:
        print("\n❌ No GPU detected. Cannot run GPU tests.")
        print("💡 To run GPU tests:")
        print("   1. Use Modular Platform with GPU instance")
        print("   2. Ensure CUDA drivers are installed")
        print("   3. Verify MAX can access GPU")
        return
    
    # Step 2: Test GPU compilation
    can_compile = test_gpu_compilation()
    
    if not can_compile:
        print("\n❌ GPU compilation failed. Check:")
        print("   1. MAX installation includes GPU support")
        print("   2. GPU drivers are compatible")
        print("   3. Memory is sufficient")
        return
    
    # Step 3: Benchmark GPU performance
    corpus_sizes = [1000, 2000, 5000, 10000]  # Start conservative
    
    print(f"\n🧪 Running GPU benchmarks...")
    print(f"   Corpus sizes: {corpus_sizes}")
    print(f"   Iterations per test: 5")
    print(f"   Configurations: FP32 + FP16")
    
    results = benchmark_gpu_performance(corpus_sizes, iterations=5)
    
    # Step 4: Analyze results
    analyze_gpu_results(results)
    
    # Step 5: Compare with CPU
    compare_gpu_vs_cpu(results)
    
    # Step 6: Save results
    save_gpu_results(results)
    
    print(f"\n✅ GPU testing complete!")
    print(f"📊 Check saved results for detailed analysis")

if __name__ == "__main__":
    main()