#!/usr/bin/env python3
"""
Simple HTTP server for the web interface
Serves the semantic search UI and proxies API requests
"""

import http.server
import socketserver
import json
import urllib.request
import urllib.parse
from pathlib import Path

PORT = 8080
API_BASE_URL = "http://localhost:8000"

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support and API proxy."""
    
    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests and proxy API calls."""
        if self.path.startswith('/api/'):
            # Proxy API requests
            api_path = self.path.replace('/api/', '/')
            self.proxy_request('GET', api_path)
        else:
            # Serve static files
            if self.path == '/':
                self.path = '/index.html'
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests and proxy API calls."""
        if self.path.startswith('/api/'):
            api_path = self.path.replace('/api/', '/')
            self.proxy_request('POST', api_path)
        else:
            self.send_error(404)
    
    def proxy_request(self, method, path):
        """Proxy requests to the API server."""
        try:
            url = f"{API_BASE_URL}{path}"
            
            # Read request body if POST
            body = None
            if method == 'POST':
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
            
            # Make request to API
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req) as response:
                # Forward response
                self.send_response(response.getcode())
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response.read())
                
        except urllib.error.HTTPError as e:
            self.send_error(e.code, e.reason)
        except Exception as e:
            print(f"Proxy error: {e}")
            self.send_error(500, str(e))

def serve():
    """Start the web server."""
    # Change to web directory
    web_dir = Path(__file__).parent
    os.chdir(web_dir)
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🌐 Web Interface Server")
        print(f"====================")
        print(f"✅ Serving at: http://localhost:{PORT}")
        print(f"✅ API proxy: http://localhost:{PORT}/api/* → {API_BASE_URL}/*")
        print(f"")
        print(f"📋 Quick Start:")
        print(f"1. Start API server: python3 api/semantic_search_api_v2.py")
        print(f"2. Open browser: http://localhost:{PORT}")
        print(f"")
        print(f"Press Ctrl+C to stop...")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")

if __name__ == "__main__":
    import os
    serve()