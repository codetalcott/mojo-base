"""
Core data structures for Mojo Semantic Search Engine.
Uses only documented Mojo features from official documentation.
"""

struct CodeSnippet:
    """Represents a code snippet with metadata."""
    var content: String
    var file_path: String 
    var project_name: String
    var function_name: String
    var line_start: Int
    var line_end: Int
    var similarity_score: Float32
    
    fn __init__(out self, content: String, file_path: String, project_name: String):
        """Initialize a CodeSnippet with required metadata."""
        self.content = content
        self.file_path = file_path
        self.project_name = project_name
        self.function_name = ""
        self.line_start = 0
        self.line_end = 0
        self.similarity_score = 0.0
    
    fn update_similarity(inout self, score: Float32):
        """Update similarity score for ranking."""
        self.similarity_score = score

struct SearchResult:
    """Search result with scoring dimensions."""
    var snippet: CodeSnippet
    var similarity_score: Float32
    var context_relevance: Float32
    var recency_boost: Float32
    var project_relevance: Float32
    var final_score: Float32
    
    fn __init__(out self, snippet: CodeSnippet):
        """Initialize search result with base snippet."""
        self.snippet = snippet
        self.similarity_score = snippet.similarity_score
        self.context_relevance = 0.0
        self.recency_boost = 0.0
        self.project_relevance = 0.0
        self.final_score = 0.0
    
    fn calculate_final_score(inout self):
        """Calculate final ranking score using weighted combination."""
        self.final_score = (
            self.similarity_score * 0.4 +
            self.context_relevance * 0.3 +
            self.recency_boost * 0.2 +
            self.project_relevance * 0.1
        )

struct SearchContext:
    """Context information for intelligent search ranking."""
    var current_project: String
    var current_file: String
    var search_focus: String
    
    fn __init__(out self, current_project: String, current_file: String):
        """Initialize search context."""
        self.current_project = current_project
        self.current_file = current_file
        self.search_focus = "general"
    
    fn set_focus(inout self, focus: String):
        """Set search focus."""
        self.search_focus = focus

struct PerformanceTracker:
    """Track search performance metrics."""
    var total_searches: Int
    var total_search_time: Float64
    var average_results_per_search: Float32
    
    fn __init__(out self):
        self.total_searches = 0
        self.total_search_time = 0.0
        self.average_results_per_search = 0.0
    
    fn record_search(inout self, search_time: Float64, num_results: Int):
        """Record a search operation."""
        self.total_searches += 1
        self.total_search_time += search_time
        
        # Update rolling average
        var alpha: Float32 = 0.1
        self.average_results_per_search = (
            (1.0 - alpha) * self.average_results_per_search +
            alpha * Float32(num_results)
        )
    
    fn get_average_time(self) -> Float64:
        """Get average search time."""
        if self.total_searches > 0:
            return self.total_search_time / Float64(self.total_searches)
        return 0.0

# Hash function for production use
fn simple_hash(s: String) -> Int:
    """Simple hash function for strings."""
    var hash_value = 0
    var s_len = len(s)
    for i in range(s_len):
        hash_value = hash_value * 31 + ord(s[i])
    if hash_value < 0:
        hash_value = -hash_value
    return hash_value

# Validation functions with proper error handling
fn validate_embedding_dimension(dimension: Int) raises:
    """Validate embedding dimension is correct."""
    if dimension != 768:
        raise Error("Embedding must be 768 dimensions")

fn validate_positive_size(size: Int, name: String) raises:
    """Validate size parameter is positive."""
    if size <= 0:
        raise Error(name + " must be positive")

fn validate_sequence_length(seq_len: Int, max_len: Int) raises:
    """Validate sequence length is within bounds."""
    if seq_len <= 0:
        raise Error("Sequence length must be positive")
    if seq_len > max_len:
        raise Error("Sequence length exceeds maximum allowed")

# Safe bounds checking functions
fn check_index_bounds(index: Int, size: Int) -> Bool:
    """Check if index is within bounds."""
    return index >= 0 and index < size

fn validate_matrix_dimensions(rows: Int, cols: Int) raises:
    """Validate matrix dimensions."""
    if rows <= 0 or cols <= 0:
        raise Error("Matrix dimensions must be positive")
    if rows > 100000 or cols > 100000:
        raise Error("Matrix dimensions exceed practical limits")