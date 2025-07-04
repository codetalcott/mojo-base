#!/usr/bin/env python3
"""
Test Demo Compatibility
Verify that updated demo scripts work with current API implementation.
"""

import sys
import time
import requests
from pathlib import Path

def test_api_endpoints():
    """Test if API endpoints match demo expectations."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing API Compatibility")
    print("=" * 40)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=3)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ /health endpoint: Available")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   MCP latency: {health_data.get('mcp_latency_ms', 'unknown')}ms")
        else:
            print("❌ /health endpoint: Error")
    except Exception as e:
        print(f"❌ /health endpoint: Failed ({e})")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=3)
        if response.status_code == 200:
            root_data = response.json()
            print("✅ / endpoint: Available")
            print(f"   Version: {root_data.get('version', 'unknown')}")
            print(f"   Corpus size: {root_data.get('corpus_size', 'unknown')}")
            print(f"   Projects: {root_data.get('source_projects', 'unknown')}")
        else:
            print("❌ / endpoint: Error")
    except Exception as e:
        print(f"❌ / endpoint: Failed ({e})")
    
    # Test search endpoint
    try:
        search_request = {
            "query": "test query",
            "max_results": 5,
            "include_mcp": True,
            "use_cache": True
        }
        response = requests.post(f"{base_url}/search", json=search_request, timeout=5)
        if response.status_code == 200:
            search_data = response.json()
            print("✅ /search endpoint: Available")
            print(f"   Results: {len(search_data.get('results', []))}")
            print(f"   Search time: {search_data.get('search_time_ms', 'unknown')}ms")
            print(f"   Performance metrics: {'✅' if 'performance_metrics' in search_data else '❌'}")
        else:
            print(f"❌ /search endpoint: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ /search endpoint: Failed ({e})")

def test_demo_imports():
    """Test if demo scripts can be imported."""
    print("\n🔧 Testing Demo Script Imports")
    print("=" * 40)
    
    demo_dir = Path(__file__).parent.parent / "demo"
    sys.path.append(str(demo_dir))
    
    # Test performance validation
    try:
        from performance_validation import Benchmark
        print("✅ performance_validation.py: Importable")
        
        # Test initialization
        benchmark = Benchmark("http://localhost:8000")
        print(f"   Targets: {benchmark.targets}")
        print(f"   Queries: {len(benchmark.demo_queries)} configured")
        
    except Exception as e:
        print(f"❌ performance_validation.py: Import failed ({e})")
    
    # Test performance dashboard
    try:
        from performance_dashboard import PerformanceDashboard
        print("✅ performance_dashboard.py: Importable")
        
        # Test initialization
        dashboard = PerformanceDashboard("http://localhost:8000")
        print(f"   Test queries: {len(dashboard.test_queries)} configured")
        print(f"   Max history: {dashboard.max_history}")
        
    except Exception as e:
        print(f"❌ performance_dashboard.py: Import failed ({e})")

def test_demo_configuration():
    """Test demo configuration matches current capabilities."""
    print("\n⚙️  Testing Demo Configuration")
    print("=" * 40)
    
    try:
        sys.path.append(str(Path(__file__).parent.parent / "demo"))
        from performance_validation import Benchmark
        
        benchmark = Benchmark()
        
        # Check if targets are reasonable
        targets = benchmark.targets
        print(f"📊 Performance Targets:")
        print(f"   Max latency: {targets['max_latency_ms']}ms")
        print(f"   Avg latency: {targets['avg_latency_ms']}ms")
        print(f"   Min results: {targets['min_results']}")
        print(f"   Min similarity: {targets['min_similarity']}")
        print(f"   Success rate: {targets['success_rate']:.1%}")
        
        # Validate targets are achievable
        reasonable_targets = (
            targets['max_latency_ms'] >= 20 and
            targets['avg_latency_ms'] >= 10 and
            targets['min_results'] >= 1 and
            targets['min_similarity'] >= 0.5 and
            targets['success_rate'] >= 0.80
        )
        
        print(f"\n✅ Targets are {'reasonable' if reasonable_targets else 'potentially unrealistic'}")
        
    except Exception as e:
        print(f"❌ Configuration test failed ({e})")

def main():
    """Run all compatibility tests."""
    print("🚀 Demo Compatibility Test Suite")
    print("=" * 60)
    print("Testing updated demo scripts against current API implementation")
    print()
    
    test_api_endpoints()
    test_demo_imports()
    test_demo_configuration()
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    print("   1. API endpoints should be available if server is running")
    print("   2. Demo scripts should import without errors")
    print("   3. Configuration should be reasonable for current performance")
    print()
    print("💡 To start API server: python api/semantic_search_api_v2.py")
    print("💡 To run demos: python demo/performance_validation.py")

if __name__ == "__main__":
    main()