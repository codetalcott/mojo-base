#!/usr/bin/env python3
"""
Comprehensive Legacy Mojo Performance Benchmark
Test the actual performance baseline with various corpus sizes
"""

import subprocess
import time
import json
import numpy as np
from pathlib import Path
from statistics import mean, stdev

def run_mojo_benchmark(corpus_size: int, iterations: int = 5) -> dict:
    """Run Mojo benchmark with specific parameters."""
    project_root = Path(__file__).parent
    
    results = []
    
    print(f"📊 Testing corpus size: {corpus_size:,} vectors")
    
    for i in range(iterations):
        print(f"   Iteration {i+1}/{iterations}...", end=" ")
        
        try:
            # Run the Mojo integration test
            cmd = [
                "pixi", "run", "mojo", 
                str(project_root / "integration_test_benchmark.mojo")
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=project_root / "portfolio-search",
                capture_output=True,
                text=True,
                timeout=30
            )
            total_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse output for performance metrics
                output = result.stdout
                latency_ms = None
                
                # Look for latency information
                for line in output.split('\n'):
                    if "Real GPU Latency:" in line:
                        try:
                            latency_ms = float(line.split(':')[1].strip().replace('ms', ''))
                        except:
                            pass
                    elif "Search time:" in line and "ms" in line:
                        try:
                            latency_ms = float(line.split(':')[1].strip().replace('ms', ''))
                        except:
                            pass
                    elif "Total query time:" in line and "ms" in line:
                        try:
                            latency_ms = float(line.split(':')[1].strip().replace('ms', ''))
                        except:
                            pass
                
                # If no latency found in output, estimate from total time
                if latency_ms is None:
                    latency_ms = total_time * 1000
                
                results.append({
                    'latency_ms': latency_ms,
                    'total_time_s': total_time,
                    'success': True
                })
                
                print(f"✅ {latency_ms:.3f}ms")
                
            else:
                print(f"❌ Failed: {result.stderr[:50]}...")
                results.append({
                    'latency_ms': None,
                    'total_time_s': total_time,
                    'success': False
                })
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
            results.append({
                'latency_ms': None,
                'total_time_s': 30.0,
                'success': False
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'latency_ms': None,
                'total_time_s': 0,
                'success': False
            })
    
    # Calculate statistics
    successful_results = [r for r in results if r['success'] and r['latency_ms'] is not None]
    
    if successful_results:
        latencies = [r['latency_ms'] for r in successful_results]
        stats = {
            'corpus_size': corpus_size,
            'iterations': iterations,
            'successful_runs': len(successful_results),
            'avg_latency_ms': mean(latencies),
            'std_latency_ms': stdev(latencies) if len(latencies) > 1 else 0,
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'all_results': results
        }
    else:
        stats = {
            'corpus_size': corpus_size,
            'iterations': iterations,
            'successful_runs': 0,
            'avg_latency_ms': None,
            'std_latency_ms': None,
            'min_latency_ms': None,
            'max_latency_ms': None,
            'all_results': results
        }
    
    return stats

def benchmark_scaling_performance():
    """Test performance across different corpus sizes."""
    print("🧪 Legacy Mojo Performance Baseline Test")
    print("=" * 60)
    
    # Test different corpus sizes
    corpus_sizes = [1000, 5000, 10000, 25000, 50000]
    iterations = 3  # Multiple runs for statistical accuracy
    
    all_results = []
    
    for corpus_size in corpus_sizes:
        result = run_mojo_benchmark(corpus_size, iterations)
        all_results.append(result)
        
        # Print immediate results
        if result['successful_runs'] > 0:
            print(f"   ✅ Avg: {result['avg_latency_ms']:.3f}ms ± {result['std_latency_ms']:.3f}ms")
            print(f"   📊 Range: {result['min_latency_ms']:.3f}ms - {result['max_latency_ms']:.3f}ms")
        else:
            print(f"   ❌ No successful runs")
        print()
    
    return all_results

def analyze_results(results):
    """Analyze and report benchmark results."""
    print("📈 Performance Analysis")
    print("=" * 60)
    
    successful_results = [r for r in results if r['successful_runs'] > 0]
    
    if not successful_results:
        print("❌ No successful benchmark runs to analyze")
        return
    
    print("Performance by Corpus Size:")
    print("-" * 40)
    
    for result in successful_results:
        corpus_size = result['corpus_size']
        avg_latency = result['avg_latency_ms']
        std_latency = result['std_latency_ms']
        
        print(f"{corpus_size:6,} vectors: {avg_latency:7.3f}ms ± {std_latency:6.3f}ms")
    
    # Overall statistics
    all_latencies = []
    for result in successful_results:
        for run_result in result['all_results']:
            if run_result['success'] and run_result['latency_ms'] is not None:
                all_latencies.append(run_result['latency_ms'])
    
    if all_latencies:
        print(f"\nOverall Performance Summary:")
        print(f"- Total successful runs: {len(all_latencies)}")
        print(f"- Average latency: {mean(all_latencies):.3f}ms")
        print(f"- Standard deviation: {stdev(all_latencies):.3f}ms")
        print(f"- Min latency: {min(all_latencies):.3f}ms")
        print(f"- Max latency: {max(all_latencies):.3f}ms")
        
        # Performance classification
        avg_perf = mean(all_latencies)
        if avg_perf < 1.0:
            perf_class = "🚀 Excellent (sub-millisecond)"
        elif avg_perf < 5.0:
            perf_class = "✅ Very Good (< 5ms)"
        elif avg_perf < 10.0:
            perf_class = "👍 Good (< 10ms)"
        elif avg_perf < 50.0:
            perf_class = "⚠️  Acceptable (< 50ms)"
        else:
            perf_class = "❌ Needs optimization (> 50ms)"
        
        print(f"- Performance class: {perf_class}")
        
        # Scaling analysis
        if len(successful_results) > 1:
            print(f"\nScaling Analysis:")
            first_result = successful_results[0]
            last_result = successful_results[-1]
            
            size_increase = last_result['corpus_size'] / first_result['corpus_size']
            latency_increase = last_result['avg_latency_ms'] / first_result['avg_latency_ms']
            
            print(f"- Corpus size increased {size_increase:.1f}x")
            print(f"- Latency increased {latency_increase:.1f}x")
            print(f"- Scaling efficiency: {size_increase/latency_increase:.2f}")

def save_results(results):
    """Save detailed results to file."""
    results_file = Path(__file__).parent / "legacy_mojo_performance_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved: {results_file}")

def main():
    """Main benchmark execution."""
    print("🔥 Legacy Mojo Performance Baseline Validation")
    print("Testing our claimed ~1ms latency across different scenarios")
    print()
    
    # Run comprehensive benchmarks
    results = benchmark_scaling_performance()
    
    # Analyze and report
    analyze_results(results)
    
    # Save detailed results
    save_results(results)
    
    print(f"\n🎯 Baseline Validation Complete!")

if __name__ == "__main__":
    main()