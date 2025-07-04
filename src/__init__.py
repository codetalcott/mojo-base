"""
Mojo Semantic Search Package
High-performance semantic search with MAX Graph optimization and Mojo kernels.
"""

# Core exports for easy cross-project usage
try:
    from .max_graph.semantic_search_graph import (
        MaxGraphConfig,
        MaxSemanticSearchGraph,
        MaxSemanticSearchBenchmark,
        create_test_data
    )
    MAX_GRAPH_AVAILABLE = True
except ImportError as e:
    # MAX dependencies not available - create placeholder classes
    MAX_GRAPH_AVAILABLE = False
    
    class MaxGraphConfig:
        """Placeholder config when MAX not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MAX Graph not available: {e}")
    
    class MaxSemanticSearchGraph:
        """Placeholder search graph when MAX not available.""" 
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MAX Graph not available: {e}")
    
    class MaxSemanticSearchBenchmark:
        """Placeholder benchmark when MAX not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MAX Graph not available: {e}")
    
    def create_test_data(*args, **kwargs):
        """Placeholder test data when MAX not available."""
        raise ImportError(f"MAX Graph not available: {e}")

try:
    from .integration.mcp_optimized_bridge import MCPOptimizedBridge
    MCP_BRIDGE_AVAILABLE = True
except ImportError:
    MCP_BRIDGE_AVAILABLE = False

# Version info
__version__ = "1.0.0"
__author__ = "Mojo Semantic Search Team"

# Main exports
__all__ = [
    "MaxGraphConfig",
    "MaxSemanticSearchGraph", 
    "MaxSemanticSearchBenchmark",
    "create_test_data",
    "MCPOptimizedBridge",
    "MAX_GRAPH_AVAILABLE",
    "MCP_BRIDGE_AVAILABLE"
]

# Convenience function for quick setup
def create_search_engine(corpus_size: int, device: str = "cpu", **kwargs):
    """
    Create a semantic search engine with sensible defaults.
    
    Args:
        corpus_size: Number of vectors in corpus
        device: Device to run on ("cpu", "gpu", "metal", etc.)
        **kwargs: Additional configuration options
        
    Returns:
        Configured MaxSemanticSearchGraph instance
        
    Example:
        >>> from src import create_search_engine
        >>> engine = create_search_engine(5000, device="cpu")
        >>> engine.compile()
        >>> results = engine.search_similarity(query, corpus)
    """
    if not MAX_GRAPH_AVAILABLE:
        raise ImportError("MAX Graph not available. Install MAX dependencies.")
    
    config = MaxGraphConfig(
        corpus_size=corpus_size,
        device=device,
        **kwargs
    )
    
    return MaxSemanticSearchGraph(config)