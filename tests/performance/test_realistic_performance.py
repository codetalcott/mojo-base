#!/usr/bin/env python3
"""
Realistic Performance Baseline Test
Test actual Mojo kernel performance, not simulation artifacts
"""

import subprocess
import time
import json
import re
from pathlib import Path
from statistics import mean, stdev

def run_semantic_search_mvp(iterations: int = 5) -> dict:
    """Run the actual semantic search MVP to get realistic performance."""
    project_root = Path(__file__).parent
    
    results = []
    
    print(f"🔥 Testing Semantic Search MVP Performance")
    print(f"   Running {iterations} iterations for statistical accuracy...")
    
    for i in range(iterations):
        print(f"   Iteration {i+1}/{iterations}...", end=" ")
        
        try:
            # Run the actual semantic search MVP
            cmd = ["pixi", "run", "mojo", str(project_root / "semantic_search_mvp.mojo")]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=project_root / "portfolio-search",
                capture_output=True,
                text=True,
                timeout=60  # Longer timeout for real computation
            )
            total_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse the actual performance metrics from output
                output = result.stdout
                
                # Extract performance metrics using regex
                embedding_time = None
                search_time = None
                total_query_time = None
                
                for line in output.split('\n'):
                    if "Query embedding:" in line and "ms" in line:
                        match = re.search(r'(\d+\.?\d*)ms', line)
                        if match:
                            embedding_time = float(match.group(1))
                    
                    elif "Similarity search:" in line and "ms" in line:
                        match = re.search(r'(\d+\.?\d*)ms', line)
                        if match:
                            search_time = float(match.group(1))
                    
                    elif "Total query time:" in line and "ms" in line:
                        match = re.search(r'(\d+\.?\d*)ms', line)
                        if match:
                            total_query_time = float(match.group(1))
                
                results.append({
                    'embedding_time_ms': embedding_time,
                    'search_time_ms': search_time,
                    'total_query_time_ms': total_query_time,
                    'total_execution_time_s': total_time,
                    'success': True
                })
                
                print(f"✅ Total: {total_query_time or total_time*1000:.1f}ms")
                
            else:
                print(f"❌ Failed: {result.stderr[:50]}...")
                results.append({
                    'embedding_time_ms': None,
                    'search_time_ms': None,
                    'total_query_time_ms': None,
                    'total_execution_time_s': total_time,
                    'success': False
                })
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
            results.append({
                'embedding_time_ms': None,
                'search_time_ms': None,
                'total_query_time_ms': None,
                'total_execution_time_s': 60.0,
                'success': False
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'embedding_time_ms': None,
                'search_time_ms': None,
                'total_query_time_ms': None,
                'total_execution_time_s': 0,
                'success': False
            })
    
    return results

def test_optimized_kernels():
    """Test the actual optimized Mojo kernels."""
    project_root = Path(__file__).parent
    
    print(f"\n🚀 Testing Optimized Mojo Kernels")
    
    # Test optimized MLA kernel
    print("   Testing optimized MLA kernel...")
    try:
        cmd = ["pixi", "run", "mojo", str(project_root / "src/kernels/mla_kernel_optimized.mojo")]
        result = subprocess.run(
            cmd,
            cwd=project_root / "portfolio-search",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("   ✅ MLA kernel working")
        else:
            print(f"   ❌ MLA kernel failed: {result.stderr[:50]}...")
    except Exception as e:
        print(f"   ❌ MLA kernel error: {e}")
    
    # Test optimized BMM kernel
    print("   Testing optimized BMM kernel...")
    try:
        cmd = ["pixi", "run", "mojo", str(project_root / "src/kernels/bmm_kernel_optimized.mojo")]
        result = subprocess.run(
            cmd,
            cwd=project_root / "portfolio-search", 
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("   ✅ BMM kernel working")
        else:
            print(f"   ❌ BMM kernel failed: {result.stderr[:50]}...")
    except Exception as e:
        print(f"   ❌ BMM kernel error: {e}")

def analyze_realistic_performance(results):
    """Analyze realistic performance metrics."""
    print(f"\n📊 Realistic Performance Analysis")
    print("=" * 50)
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful runs to analyze")
        return
    
    # Analyze each metric
    metrics = ['embedding_time_ms', 'search_time_ms', 'total_query_time_ms']
    
    for metric in metrics:
        values = [r[metric] for r in successful_results if r[metric] is not None]
        
        if values:
            avg_val = mean(values)
            std_val = stdev(values) if len(values) > 1 else 0
            min_val = min(values)
            max_val = max(values)
            
            metric_name = metric.replace('_', ' ').replace('ms', '(ms)').title()
            print(f"{metric_name:20}: {avg_val:6.1f}ms ± {std_val:5.1f}ms (range: {min_val:.1f}-{max_val:.1f}ms)")
    
    # Overall assessment
    total_times = [r['total_query_time_ms'] for r in successful_results if r['total_query_time_ms'] is not None]
    
    if total_times:
        avg_total = mean(total_times)
        
        print(f"\n🎯 Performance Assessment:")
        print(f"   Average total latency: {avg_total:.1f}ms")
        
        if avg_total < 10:
            assessment = "🚀 Excellent (< 10ms)"
        elif avg_total < 20:
            assessment = "✅ Very Good (< 20ms)"
        elif avg_total < 50:
            assessment = "👍 Good (< 50ms - real-time capable)"
        elif avg_total < 100:
            assessment = "⚠️  Acceptable (< 100ms)"
        else:
            assessment = "❌ Needs optimization (> 100ms)"
        
        print(f"   Classification: {assessment}")
        
        # Compare to our original claim
        if avg_total < 2:
            print(f"   ✅ Better than claimed ~1ms baseline!")
        elif avg_total < 20:
            print(f"   ✅ Meets real-time performance requirements")
        else:
            print(f"   ⚠️  Performance higher than expected, but still acceptable")

def main():
    """Main realistic performance test."""
    print("🧪 Realistic Mojo Performance Baseline Test")
    print("=" * 60)
    print("Testing actual semantic search performance, not simulation artifacts")
    print()
    
    # Test optimized kernels
    test_optimized_kernels()
    
    # Run realistic performance test
    results = run_semantic_search_mvp(iterations=3)
    
    # Analyze results
    analyze_realistic_performance(results)
    
    # Save results
    results_file = Path(__file__).parent / "realistic_performance_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'test_type': 'realistic_semantic_search_performance',
            'results': results,
            'summary': {
                'successful_runs': len([r for r in results if r['success']]),
                'total_runs': len(results)
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    print(f"\n🔍 Baseline Performance Reality Check Complete!")

if __name__ == "__main__":
    main()