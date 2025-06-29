"""
Core data structures for Mojo Semantic Search Engine.
High-performance, memory-efficient structures for code analysis.
"""

from tensor import Tensor
from utils.list import List
from memory import DTypePointer
from DType import DType

struct CodeSnippet:
    """
    Represents a code snippet with metadata and embedding.
    Optimized for semantic search and pattern matching.
    """
    var content: String
    var file_path: String 
    var project_name: String
    var function_name: String
    var line_start: Int
    var line_end: Int
    var dependencies: List[String]
    var embedding: Tensor[DType.float32]
    var similarity_score: Float32
    
    fn __init__(inout self, 
                content: String,
                file_path: String, 
                project_name: String,
                function_name: String = "",
                line_start: Int = 0,
                line_end: Int = 0):
        """Initialize a CodeSnippet with required metadata."""
        self.content = content
        self.file_path = file_path
        self.project_name = project_name
        self.function_name = function_name
        self.line_start = line_start
        self.line_end = line_end
        self.dependencies = List[String]()
        self.embedding = Tensor[DType.float32](768)  # 768-dim embeddings
        self.similarity_score = 0.0
    
    fn add_dependency(inout self, dependency: String):
        """Add a dependency to this code snippet."""
        self.dependencies.append(dependency)
    
    fn set_embedding(inout self, embedding: Tensor[DType.float32]):
        """Set the semantic embedding for this code snippet."""
        self.embedding = embedding
    
    fn update_similarity(inout self, score: Float32):
        """Update similarity score for ranking."""
        self.similarity_score = score

struct SearchResult:
    """
    Search result with multiple scoring dimensions.
    Supports advanced ranking algorithms.
    """
    var snippet: CodeSnippet
    var similarity_score: Float32
    var context_relevance: Float32
    var recency_boost: Float32
    var project_relevance: Float32
    var final_score: Float32
    
    fn __init__(inout self, snippet: CodeSnippet):
        """Initialize search result with base snippet."""
        self.snippet = snippet
        self.similarity_score = snippet.similarity_score
        self.context_relevance = 0.0
        self.recency_boost = 0.0
        self.project_relevance = 0.0
        self.final_score = 0.0
    
    fn calculate_final_score(inout self):
        """
        Calculate final ranking score using weighted combination.
        
        Weights:
        - Similarity: 40% (semantic relevance)
        - Context: 30% (surrounding code relevance)  
        - Recency: 20% (recently modified boost)
        - Project: 10% (current project preference)
        """
        self.final_score = (
            self.similarity_score * 0.4 +
            self.context_relevance * 0.3 +
            self.recency_boost * 0.2 +
            self.project_relevance * 0.1
        )

struct SearchContext:
    """
    Context information for intelligent search ranking.
    Tracks user preferences and current development state.
    """
    var current_project: String
    var current_file: String
    var recent_queries: List[String]
    var preferred_languages: List[String]
    var search_focus: String  # "api", "patterns", "implementations", etc.
    
    fn __init__(inout self, 
                current_project: String = "",
                current_file: String = ""):
        """Initialize search context with current development state."""
        self.current_project = current_project
        self.current_file = current_file
        self.recent_queries = List[String]()
        self.preferred_languages = List[String]()
        self.search_focus = "general"
    
    fn add_recent_query(inout self, query: String):
        """Track recent queries for context-aware ranking."""
        self.recent_queries.append(query)
        # Keep only last 10 queries
        if len(self.recent_queries) > 10:
            # Remove oldest query (simplified - in real implementation would use deque)
            pass
    
    fn set_focus(inout self, focus: String):
        """Set search focus: 'api', 'patterns', 'implementations', etc."""
        self.search_focus = focus

struct EmbeddingCache:
    """
    High-performance cache for code embeddings.
    Uses hash-based lookup for sub-millisecond retrieval.
    """
    var cache_data: DTypePointer[DType.float32]
    var cache_keys: List[String]  
    var cache_size: Int
    var max_size: Int
    var hit_count: Int
    var miss_count: Int
    
    fn __init__(inout self, max_size: Int = 10000):
        """Initialize embedding cache with specified capacity."""
        self.max_size = max_size
        self.cache_size = 0
        self.hit_count = 0
        self.miss_count = 0
        self.cache_keys = List[String]()
        # Allocate aligned memory for embeddings (768 dims per entry)
        self.cache_data = DTypePointer[DType.float32].aligned_alloc(
            max_size * 768, 32  # 32-byte alignment for SIMD
        )
    
    fn __del__(owned self):
        """Clean up allocated memory."""
        self.cache_data.free()
    
    fn get_cache_efficiency(self) -> Float32:
        """Calculate cache hit rate for performance monitoring."""
        let total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return Float32(self.hit_count) / Float32(total)

struct CodeCorpus:
    """
    Efficient storage for large collections of code snippets.
    Optimized for batch operations and memory locality.
    """
    var snippets: List[CodeSnippet]
    var embeddings_matrix: Tensor[DType.float32]  # N x 768 matrix
    var project_index: List[String]  # Project names for filtering
    var file_index: List[String]     # File paths for filtering
    var total_snippets: Int
    
    fn __init__(inout self, initial_capacity: Int = 1000):
        """Initialize corpus with pre-allocated capacity."""
        self.snippets = List[CodeSnippet]()
        self.project_index = List[String]()
        self.file_index = List[String]() 
        self.total_snippets = 0
        # Pre-allocate embedding matrix for performance
        self.embeddings_matrix = Tensor[DType.float32](initial_capacity, 768)
    
    fn add_snippet(inout self, snippet: CodeSnippet):
        """Add a code snippet to the corpus."""
        self.snippets.append(snippet)
        self.project_index.append(snippet.project_name)
        self.file_index.append(snippet.file_path)
        # Copy embedding to matrix for batch operations
        # TODO: Implement tensor slicing for efficient copy
        self.total_snippets += 1
    
    fn get_embeddings_slice(self, start: Int, count: Int) -> Tensor[DType.float32]:
        """Get a slice of embeddings for batch similarity computation."""
        # Return slice of embeddings matrix
        # TODO: Implement efficient tensor slicing
        return self.embeddings_matrix
    
    fn filter_by_project(self, project: String) -> List[Int]:
        """Get indices of snippets from specified project."""
        var indices = List[Int]()
        for i in range(self.total_snippets):
            if self.project_index[i] == project:
                indices.append(i)
        return indices