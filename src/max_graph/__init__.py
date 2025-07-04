"""
MAX Graph Semantic Search Implementation
Optimized semantic search using Modular's MAX Graph API.
"""

try:
    from .semantic_search_graph import (
        MaxGraphConfig,
        MaxSemanticSearchGraph,
        MaxSemanticSearchBenchmark,
        create_test_data
    )
    MAX_AVAILABLE = True
except ImportError as import_error:
    # MAX dependencies not available - create placeholder classes
    MAX_AVAILABLE = False
    _error_msg = str(import_error)
    
    class MaxGraphConfig:
        """Placeholder config when MAX not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MAX Graph not available: {_error_msg}")
    
    class MaxSemanticSearchGraph:
        """Placeholder search graph when MAX not available.""" 
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MAX Graph not available: {_error_msg}")
    
    class MaxSemanticSearchBenchmark:
        """Placeholder benchmark when MAX not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"MAX Graph not available: {_error_msg}")
    
    def create_test_data(*args, **kwargs):
        """Placeholder test data when MAX not available."""
        raise ImportError(f"MAX Graph not available: {_error_msg}")

__all__ = [
    "MaxGraphConfig",
    "MaxSemanticSearchGraph", 
    "MaxSemanticSearchBenchmark",
    "create_test_data",
    "MAX_AVAILABLE"
]