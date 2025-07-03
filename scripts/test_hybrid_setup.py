#!/usr/bin/env python3
"""
Test Hybrid Setup - Verify MAX Graph + Legacy Mojo Integration
Quick test to ensure both implementations work before full autotuning
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_max_graph():
    """Test MAX Graph implementation."""
    print("🚀 Testing MAX Graph Implementation")
    print("-" * 40)
    
    try:
        from src.max_graph.semantic_search_graph import (
            MaxSemanticSearchGraph, 
            MaxGraphConfig,
            create_test_data
        )
        
        # Create test configuration
        config = MaxGraphConfig(
            corpus_size=1000,  # Small for quick test
            vector_dims=768,
            device="cpu",
            use_fp16=False,
            enable_fusion=True
        )
        
        print(f"   Config: {config.corpus_size} vectors, {config.vector_dims}D")
        print(f"   Device: {config.device}, FP16: {config.use_fp16}")
        
        # Create test data
        query_embeddings, corpus_embeddings = create_test_data(
            config.corpus_size, config.vector_dims
        )
        print(f"   Test data: {query_embeddings.shape}, {corpus_embeddings.shape}")
        
        # Initialize MAX Graph
        start_time = time.time()
        max_search = MaxSemanticSearchGraph(config)
        max_search.compile()
        compile_time = time.time() - start_time
        
        print(f"   Compilation: {compile_time:.3f}s")
        
        # Test search
        start_time = time.time()
        result = max_search.search_similarity(query_embeddings[0], corpus_embeddings)
        search_time = time.time() - start_time
        
        print(f"   Search time: {search_time * 1000:.3f}ms")
        print(f"   Result shape: {result['similarities'].shape}")
        
        # Get top results
        top_indices, top_scores = max_search.get_top_k_results(result['similarities'], k=5)
        print(f"   Top 5 scores: {top_scores}")
        
        print("✅ MAX Graph test successful!")
        return True, {
            'latency_ms': search_time * 1000,
            'compilation_time': compile_time,
            'top_scores': top_scores.tolist()
        }
        
    except ImportError as e:
        print(f"❌ MAX Graph import failed: {e}")
        return False, {'error': str(e)}
    except Exception as e:
        print(f"❌ MAX Graph test failed: {e}")
        return False, {'error': str(e)}

def test_legacy_mojo():
    """Test legacy Mojo implementation via integration test."""
    print("\n🔥 Testing Legacy Mojo Implementation")
    print("-" * 40)
    
    try:
        import subprocess
        
        # Run quick integration test
        cmd = ["pixi", "run", "mojo", str(project_root / "integration_test_benchmark.mojo")]
        
        print("   Running integration test benchmark...")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            cwd=project_root / "portfolio-search",
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout for quick test
        )
        
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            # Parse output for performance metrics
            output = result.stdout
            latency_ms = 1.0  # Default
            
            for line in output.split('\n'):
                if "Real GPU Latency:" in line:
                    try:
                        latency_ms = float(line.split(':')[1].strip().replace('ms', ''))
                    except:
                        pass
            
            print(f"   Execution time: {total_time:.3f}s")
            print(f"   Reported latency: {latency_ms:.3f}ms")
            print("✅ Legacy Mojo test successful!")
            
            return True, {
                'latency_ms': latency_ms,
                'total_time': total_time,
                'output_lines': len(output.split('\n'))
            }
        else:
            print(f"❌ Legacy Mojo test failed:")
            print(f"   Return code: {result.returncode}")
            print(f"   Error: {result.stderr[:200]}...")
            return False, {'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print("❌ Legacy Mojo test timed out")
        return False, {'error': 'timeout'}
    except Exception as e:
        print(f"❌ Legacy Mojo test failed: {e}")
        return False, {'error': str(e)}

def compare_implementations(max_result, legacy_result):
    """Compare results from both implementations."""
    print("\n📊 Implementation Comparison")
    print("-" * 40)
    
    if max_result[0] and legacy_result[0]:
        max_latency = max_result[1]['latency_ms']
        legacy_latency = legacy_result[1]['latency_ms']
        
        speedup = legacy_latency / max_latency if max_latency > 0 else 1.0
        
        print(f"   MAX Graph latency: {max_latency:.3f}ms")
        print(f"   Legacy Mojo latency: {legacy_latency:.3f}ms")
        print(f"   Speedup factor: {speedup:.2f}x")
        
        if speedup > 1.1:
            winner = "MAX Graph"
            improvement = (speedup - 1) * 100
            print(f"🏆 Winner: {winner} ({improvement:.1f}% faster)")
        elif speedup < 0.9:
            winner = "Legacy Mojo"
            improvement = (1/speedup - 1) * 100
            print(f"🏆 Winner: {winner} ({improvement:.1f}% faster)")
        else:
            winner = "Similar performance"
            print(f"🤝 Result: {winner}")
        
        return winner, speedup
        
    elif max_result[0]:
        print("✅ Only MAX Graph working")
        return "MAX Graph (only option)", 1.0
        
    elif legacy_result[0]:
        print("✅ Only Legacy Mojo working")
        return "Legacy Mojo (only option)", 1.0
        
    else:
        print("❌ Neither implementation working")
        return "None working", 0.0

def main():
    """Run hybrid setup test."""
    print("🧪 Hybrid Setup Test - MAX Graph + Legacy Mojo")
    print("=" * 60)
    print("Testing both implementations before full autotuning...")
    
    # Test MAX Graph
    max_success, max_metrics = test_max_graph()
    
    # Test Legacy Mojo
    legacy_success, legacy_metrics = test_legacy_mojo()
    
    # Compare results
    winner, speedup = compare_implementations(
        (max_success, max_metrics), 
        (legacy_success, legacy_metrics)
    )
    
    # Summary
    print(f"\n🎯 Test Summary")
    print("=" * 60)
    print(f"MAX Graph available: {'✅ Yes' if max_success else '❌ No'}")
    print(f"Legacy Mojo available: {'✅ Yes' if legacy_success else '❌ No'}")
    print(f"Performance winner: {winner}")
    
    if max_success or legacy_success:
        print(f"\n✅ Hybrid setup ready for full autotuning!")
        print(f"   Run: python scripts/autotuning_v2_max_hybrid.py")
    else:
        print(f"\n❌ Setup issues detected - need to fix before autotuning")
        
        if not max_success:
            print(f"   MAX Graph issue: {max_metrics.get('error', 'unknown')}")
        if not legacy_success:
            print(f"   Legacy Mojo issue: {legacy_metrics.get('error', 'unknown')}")

if __name__ == "__main__":
    main()