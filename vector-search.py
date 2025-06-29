#!/usr/bin/env python3
"""
Vector Search Interface for Mojo-Base using Onedev
"""
import sys
import os

# Add onedev project to path for direct tool access
sys.path.insert(0, '<onedev-project-path>/src')

def semantic_search(query: str):
    """Perform semantic search using onedev vector capabilities"""
    print(f"🔍 Searching for: '{query}'")
    print("=" * 50)
    
    # Use onedev's existing vector search via MCP
    os.system(f"""
    cd <project-root> && 
    node -e "
    const {{ spawn }} = require('child_process');
    const mcp = spawn('node', [
        '<onedev-project-path>/dist/infrastructure/mcp/unified-mcp-main-v2.js'
    ]);
    
    // Send semantic search request
    const request = {{
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/call',
        params: {{
            name: 'search_codebase_knowledge',
            arguments: {{
                query: '{query}',
                project_path: '<project-root>'
            }}
        }}
    }};
    
    mcp.stdin.write(JSON.stringify(request) + '\\n');
    
    mcp.stdout.on('data', (data) => {{
        console.log('Search Results:', data.toString());
    }});
    
    setTimeout(() => mcp.kill(), 5000);
    "
    """)

def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python vector-search.py '<search query>'")
        print("Example: python vector-search.py 'mojo kernel implementation'")
        return
    
    query = " ".join(sys.argv[1:])
    semantic_search(query)

if __name__ == "__main__":
    main()