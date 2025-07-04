"""
Integration Layer for External Systems
Bridges for MCP, vector databases, and other external services.
"""

try:
    from .mcp_optimized_bridge import MCPOptimizedBridge
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    from .vector_db_analyzer import VectorDBAnalyzer
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False

__all__ = [
    "MCPOptimizedBridge",
    "VectorDBAnalyzer", 
    "MCP_AVAILABLE",
    "VECTOR_DB_AVAILABLE"
]