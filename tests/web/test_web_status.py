#!/usr/bin/env python3
"""
Test the complete web interface status
"""

import requests
import json

def test_api_health():
    """Test API server health and performance."""
    print("🔍 Testing API Server")
    print("-" * 30)
    
    try:
        # Test search endpoint
        response = requests.get(
            "http://localhost:8000/search/simple",
            params={"q": "authentication patterns", "limit": 3},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: Working")
            print(f"✅ Search Time: {data.get('search_time_ms', 'unknown'):.1f}ms")
            print(f"✅ Results Found: {len(data.get('results', []))}")
            print(f"✅ Corpus Size: {data.get('corpus_size', 'unknown'):,}")
            print(f"✅ MCP Enhanced: {data.get('mcp_enhanced', False)}")
            
            # Show sample result
            if data.get('results'):
                sample = data['results'][0]
                print(f"✅ Sample Result: {sample.get('project', 'unknown')}/{sample.get('file_path', 'unknown')}")
                print(f"   Similarity: {sample.get('similarity_score', 0):.2f}")
            
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ API Connection Failed: {e}")
        return False

def test_web_interface():
    """Test web interface."""
    print(f"\n🌐 Testing Web Interface")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        
        if response.status_code == 200:
            content = response.text
            if "Mojo Semantic Search" in content:
                print("✅ Web Interface: Working")
                print("✅ HTML Content: Valid")
                
                # Check for key components
                checks = [
                    ("Search Input", "searchQuery" in content),
                    ("Results Display", "searchResults" in content),
                    ("Performance Chart", "performanceChart" in content),
                    ("Alpine.js", "alpinejs" in content),
                    ("DaisyUI", "daisyui" in content)
                ]
                
                for check_name, check_result in checks:
                    status = "✅" if check_result else "❌"
                    print(f"{status} {check_name}: {'Present' if check_result else 'Missing'}")
                
                return True
            else:
                print("❌ Web Interface: Invalid content")
                return False
        else:
            print(f"❌ Web Interface: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Web Interface Connection Failed: {e}")
        return False

def test_integration():
    """Test full integration."""
    print(f"\n🔗 Testing Full Integration")
    print("-" * 30)
    
    # This would test the API proxy, but since we're using direct API calls,
    # we'll test the workflow
    
    try:
        # Simulate a complete search workflow
        queries = ["authentication", "React components", "API patterns"]
        
        for query in queries:
            response = requests.get(
                "http://localhost:8000/search/simple",
                params={"q": query, "limit": 2},
                timeout=3
            )
            
            if response.status_code == 200:
                data = response.json()
                search_time = data.get('search_time_ms', 0)
                results_count = len(data.get('results', []))
                print(f"✅ Query '{query}': {search_time:.1f}ms, {results_count} results")
            else:
                print(f"❌ Query '{query}': Failed")
                return False
        
        print("✅ Integration: All queries successful")
        return True
        
    except Exception as e:
        print(f"❌ Integration Test Failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Web Interface Status Check")
    print("=" * 50)
    
    # Test components
    api_ok = test_api_health()
    web_ok = test_web_interface()
    integration_ok = test_integration() if api_ok else False
    
    # Summary
    print(f"\n📊 Summary")
    print("=" * 20)
    print(f"API Server: {'✅ Working' if api_ok else '❌ Failed'}")
    print(f"Web Interface: {'✅ Working' if web_ok else '❌ Failed'}")
    print(f"Integration: {'✅ Working' if integration_ok else '❌ Failed'}")
    
    if api_ok and web_ok:
        print(f"\n🎉 Web Interface Status: READY!")
        print(f"📋 Access URLs:")
        print(f"   • Web Interface: http://localhost:8080")
        print(f"   • API Direct: http://localhost:8000")
        print(f"   • API Docs: http://localhost:8000/docs")
        
        if not integration_ok:
            print(f"\n⚠️  Note: Direct API access working, web proxy may need setup")
            print(f"   Web interface should work with mock data fallback")
    else:
        print(f"\n❌ Web Interface Status: NEEDS ATTENTION")
        
        if not api_ok:
            print(f"   1. Start API: python api/semantic_search_api_v2.py")
        if not web_ok:
            print(f"   2. Start Web: python web/server.py")

if __name__ == "__main__":
    main()