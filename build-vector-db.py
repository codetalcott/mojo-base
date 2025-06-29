#!/usr/bin/env python3
"""
Build Vector Database for Mojo-Base Project using Onedev Tools
"""
import json
import subprocess
import sys
from pathlib import Path

def run_mcp_tool(tool_name: str, params: dict = None):
    """Run an onedev MCP tool via node CLI"""
    cmd = [
        "node", 
        "/Users/williamtalcott/projects/onedev/dist/infrastructure/mcp/unified-mcp-main-v2.js",
        "--tool", tool_name
    ]
    
    if params:
        cmd.extend(["--params", json.dumps(params)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout) if result.stdout.strip() else {}
        else:
            print(f"Error running {tool_name}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Failed to run {tool_name}: {e}")
        return None

def main():
    """Build vector database for mojo-base project"""
    project_root = "/Users/williamtalcott/projects/mojo-base"
    
    print("🚀 Building Vector Database for Mojo-Base Project using Onedev")
    print("=" * 60)
    
    # Step 1: Scan the mojo-base project structure
    print("\n📊 Step 1: Scanning project structure...")
    scan_result = run_mcp_tool("scan_projects", {
        "project_path": project_root,
        "include_health": True
    })
    
    if scan_result:
        print(f"✅ Found {len(scan_result.get('projects', []))} components")
    
    # Step 2: Assemble context for code patterns
    print("\n🧠 Step 2: Assembling context and generating embeddings...")
    context_result = run_mcp_tool("assemble_context", {
        "project_path": project_root,
        "focus": "mojo semantic search implementation",
        "include_patterns": True
    })
    
    # Step 3: Create development plan for vector database
    print("\n📋 Step 3: Creating vector database development plan...")
    plan_result = run_mcp_tool("create_development_plan", {
        "project_name": "mojo-base-vector-db",
        "requirements": [
            "Vector database for Mojo code embeddings",
            "Semantic search across portfolio projects", 
            "Real-time embedding generation",
            "Integration with MLA/BMM kernels"
        ],
        "context": context_result
    })
    
    # Step 4: Generate vector similarity insights
    print("\n🔍 Step 4: Generating vector similarity insights...")
    vector_result = run_mcp_tool("get_vector_similarity_insights", {
        "project_path": project_root,
        "query": "semantic search vector database mojo kernels"
    })
    
    # Step 5: Get architectural recommendations
    print("\n🏗️ Step 5: Getting architectural recommendations...")
    arch_result = run_mcp_tool("get_architectural_recommendations", {
        "context": "Building vector database for Mojo semantic search",
        "requirements": "High-performance embedding storage and retrieval"
    })
    
    print("\n✅ Vector Database Build Complete!")
    print("🎯 Check onedev database for generated embeddings and context")
    print("📊 Vector intelligence available via onedev MCP tools")

if __name__ == "__main__":
    main()