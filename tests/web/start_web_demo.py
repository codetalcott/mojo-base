#!/usr/bin/env python3
"""
Start web demo with proper CORS and API proxy
"""

import sys
import subprocess
import time
import requests
from pathlib import Path

def check_api_server():
    """Check if API server is running."""
    try:
        response = requests.get("http://localhost:8000/search/simple?q=test&limit=1", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_web_server():
    """Start the web server with proper configuration."""
    web_dir = Path(__file__).parent / "web"
    
    # Import and run the web server
    sys.path.append(str(web_dir))
    
    import server
    server.serve()

def main():
    """Main demo starter."""
    print("🌐 Starting Mojo Semantic Search Web Demo")
    print("=" * 50)
    
    # Check API server
    if check_api_server():
        print("✅ API server running at http://localhost:8000")
    else:
        print("❌ API server not running at http://localhost:8000")
        print("   Please start: python api/semantic_search_api_v2.py")
        return
    
    # Start web server
    print("🚀 Starting web server at http://localhost:8080")
    print()
    start_web_server()

if __name__ == "__main__":
    main()