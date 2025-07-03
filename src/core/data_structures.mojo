"""
Modern Core Data Structures for Mojo Semantic Search Engine.
Updated for Mojo 25.4.0 compatibility with proper syntax and copy semantics.
"""

from memory import UnsafePointer

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
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for CodeSnippet."""
        self.content = existing.content
        self.file_path = existing.file_path
        self.project_name = existing.project_name
        self.function_name = existing.function_name
        self.line_start = existing.line_start
        self.line_end = existing.line_end
        self.similarity_score = existing.similarity_score
    
    fn update_similarity(mut self, score: Float32):
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
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for SearchResult."""
        self.snippet = existing.snippet
        self.similarity_score = existing.similarity_score
        self.context_relevance = existing.context_relevance
        self.recency_boost = existing.recency_boost
        self.project_relevance = existing.project_relevance
        self.final_score = existing.final_score
    
    fn calculate_final_score(mut self):
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
    
    fn __init__(out self):
        """Initialize search context with defaults."""
        self.current_project = ""
        self.current_file = ""
        self.search_focus = "general"
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for SearchContext."""
        self.current_project = existing.current_project
        self.current_file = existing.current_file
        self.search_focus = existing.search_focus
    
    fn set_focus(mut self, focus: String):
        """Set search focus."""
        self.search_focus = focus

struct PerformanceTracker:
    """Track search performance metrics."""
    var total_searches: Int
    var total_search_time: Float64
    var average_results_per_search: Float32
    
    fn __init__(out self):
        """Initialize performance tracker."""
        self.total_searches = 0
        self.total_search_time = 0.0
        self.average_results_per_search = 0.0
    
    fn __copyinit__(out self, existing: Self):
        """Copy constructor for PerformanceTracker."""
        self.total_searches = existing.total_searches
        self.total_search_time = existing.total_search_time
        self.average_results_per_search = existing.average_results_per_search
    
    fn record_search(mut self, search_time: Float64, num_results: Int):
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

struct CodeCorpus:
    """Manages a collection of code snippets with embeddings."""
    var snippets: UnsafePointer[CodeSnippet]
    var embeddings: UnsafePointer[Float32]  # Flattened embeddings matrix
    var capacity: Int
    var size: Int
    var embed_dim: Int
    
    fn __init__(out self, capacity: Int, embed_dim: Int = 768) raises:
        """Initialize code corpus with specified capacity."""
        if capacity <= 0:
            raise Error("Corpus capacity must be positive")
        if embed_dim <= 0:
            raise Error("Embedding dimension must be positive")
        
        self.capacity = capacity
        self.size = 0
        self.embed_dim = embed_dim
        
        # Allocate memory for snippets and embeddings
        self.snippets = UnsafePointer[CodeSnippet].alloc(capacity)
        self.embeddings = UnsafePointer[Float32].alloc(capacity * embed_dim)
        
        # Initialize embeddings to zero
        for i in range(capacity * embed_dim):
            self.embeddings[i] = 0.0
    
    fn __del__(owned self):
        """Clean up allocated memory."""
        self.snippets.free()
        self.embeddings.free()
    
    fn add_snippet(mut self, snippet: CodeSnippet, embedding: UnsafePointer[Float32]) raises -> Bool:
        """Add a code snippet with its embedding to the corpus."""
        if self.size >= self.capacity:
            return False
        
        # Store snippet (using copy constructor)
        self.snippets[self.size] = snippet
        
        # Store embedding
        var offset = self.size * self.embed_dim
        for i in range(self.embed_dim):
            self.embeddings[offset + i] = embedding[i]
        
        self.size += 1
        return True
    
    fn get_snippet(self, index: Int) -> CodeSnippet:
        """Get snippet at specified index."""
        return self.snippets[index]
    
    fn get_embedding(self, index: Int, output: UnsafePointer[Float32]):
        """Get embedding at specified index."""
        var offset = index * self.embed_dim
        for i in range(self.embed_dim):
            output[i] = self.embeddings[offset + i]

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

# Test the data structures
fn test_data_structures():
    """Test the modernized data structures."""
    print("🧪 Testing Modern Data Structures")
    print("=================================")
    
    try:
        # Test CodeSnippet
        var snippet = CodeSnippet("def hello(): print('hi')", "test.py", "test_proj")
        snippet.update_similarity(0.95)
        print("✅ CodeSnippet: Created and updated similarity score")
        
        # Test SearchResult
        var result = SearchResult(snippet)
        result.context_relevance = 0.8
        result.recency_boost = 0.7
        result.project_relevance = 0.9
        result.calculate_final_score()
        print("✅ SearchResult: Created and calculated final score:", result.final_score)
        
        # Test SearchContext
        var context = SearchContext("main_project", "src/main.py")
        context.set_focus("authentication")
        print("✅ SearchContext: Created and set focus")
        
        # Test PerformanceTracker
        var tracker = PerformanceTracker()
        tracker.record_search(0.025, 15)
        tracker.record_search(0.030, 12)
        var avg_time = tracker.get_average_time()
        print("✅ PerformanceTracker: Recorded searches, avg time:", avg_time)
        
        # Test CodeCorpus
        var corpus = CodeCorpus(100, 768)
        var test_embedding = UnsafePointer[Float32].alloc(768)
        for i in range(768):
            test_embedding[i] = Float32(i) / 768.0
        
        var added = corpus.add_snippet(snippet, test_embedding)
        print("✅ CodeCorpus: Added snippet, success:", added)
        
        test_embedding.free()
        
        print("✅ All data structures working correctly!")
        
    except e:
        print("❌ Data structures test failed:", e)

fn main():
    """Test the modern data structures."""
    test_data_structures()