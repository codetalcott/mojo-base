"""
Semantic Search Engine for Cross-Project Code Search.
Integrates MLA and BMM kernels for real-time semantic understanding.
"""

from tensor import Tensor
from utils.list import List
from memory import DTypePointer
from DType import DType
from ..core.data_structures import CodeSnippet, SearchResult, SearchContext, CodeCorpus
from ..kernels.mla_kernel import MLAKernel, create_optimized_mla_kernel
from ..kernels.bmm_kernel import BMMKernel, create_bmm_kernel

struct SemanticSearchEngine:
    """
    Main semantic search engine coordinating all components.
    
    Architecture:
    - MLA kernel for query/code embedding generation
    - BMM kernel for ultra-fast similarity search
    - Advanced ranking with context awareness
    - Integration with onedev portfolio intelligence
    """
    var mla_model: MLAKernel
    var bmm_kernel: BMMKernel
    var code_corpus: CodeCorpus
    var search_context: SearchContext
    var total_embeddings: Int
    var performance_stats: PerformanceTracker
    
    fn __init__(inout self, max_corpus_size: Int = 100000):
        """Initialize semantic search engine with specified capacity."""
        self.mla_model = create_optimized_mla_kernel()
        self.bmm_kernel = create_bmm_kernel(max_corpus_size)
        self.code_corpus = CodeCorpus(max_corpus_size)
        self.search_context = SearchContext()
        self.total_embeddings = 0
        self.performance_stats = PerformanceTracker()
    
    fn index_code_snippet(inout self, snippet: CodeSnippet) -> Bool:
        """
        Index a code snippet for semantic search.
        
        Args:
            snippet: Code snippet with metadata
            
        Returns:
            True if successfully indexed, False if corpus is full
        """
        if self.total_embeddings >= self.code_corpus.embeddings_matrix.shape()[0]:
            return False
        
        # Generate embedding using MLA kernel
        let tokens = self._tokenize_code(snippet.content)
        let embedding = self.mla_model.encode_sequence(tokens, tokens.shape()[0])
        
        # Create indexed snippet with embedding
        var indexed_snippet = snippet
        indexed_snippet.set_embedding(embedding)
        
        # Add to corpus
        self.code_corpus.add_snippet(indexed_snippet)
        self.total_embeddings += 1
        
        # Update BMM kernel if needed (batch updates for efficiency)
        if self.total_embeddings % 1000 == 0:
            self._update_bmm_corpus()
        
        return True
    
    fn search(inout self, 
             query: String, 
             max_results: Int = 20,
             project_filter: String = "") -> List[SearchResult]:
        """
        Perform semantic search across indexed code.
        
        Args:
            query: Natural language or code query
            max_results: Maximum number of results to return
            project_filter: Optional project name filter
            
        Returns:
            Ranked list of search results
        """
        self.performance_stats.start_search()
        
        # Step 1: Generate query embedding
        let query_tokens = self._tokenize_query(query)
        let query_embedding = self.mla_model.encode_sequence(
            query_tokens, query_tokens.shape()[0]
        )
        
        # Step 2: Fast similarity search using BMM kernel
        let (similarity_scores, indices) = self.bmm_kernel.batched_similarity_top_k(
            query_embedding, max_results * 2  # Get more for filtering/ranking
        )
        
        # Step 3: Create search results with context-aware ranking
        var results = List[SearchResult]()
        
        for i in range(min(max_results * 2, len(indices))):
            let corpus_idx = indices[i]
            let snippet = self.code_corpus.snippets[corpus_idx]
            
            # Apply project filter if specified
            if project_filter != "" and snippet.project_name != project_filter:
                continue
            
            # Create search result with base similarity
            var result = SearchResult(snippet)
            result.similarity_score = similarity_scores[i]
            
            # Enhanced ranking with context
            self._apply_context_ranking(result)
            result.calculate_final_score()
            
            results.append(result)
            
            if len(results) >= max_results:
                break
        
        # Sort by final score
        self._sort_results_by_score(results)
        
        # Update search context
        self.search_context.add_recent_query(query)
        
        self.performance_stats.end_search(len(results))
        return results
    
    fn search_by_code_similarity(inout self,
                                code_snippet: String,
                                max_results: Int = 10) -> List[SearchResult]:
        """
        Find similar code patterns to a given code snippet.
        
        Args:
            code_snippet: Code to find similar patterns for
            max_results: Maximum results to return
            
        Returns:
            Similar code snippets ranked by semantic similarity
        """
        # Tokenize and embed the input code
        let code_tokens = self._tokenize_code(code_snippet)
        let code_embedding = self.mla_model.encode_sequence(
            code_tokens, code_tokens.shape()[0]
        )
        
        # Search using the code embedding as query
        let (similarity_scores, indices) = self.bmm_kernel.batched_similarity_top_k(
            code_embedding, max_results
        )
        
        var results = List[SearchResult]()
        
        for i in range(len(indices)):
            let corpus_idx = indices[i]
            let snippet = self.code_corpus.snippets[corpus_idx]
            
            var result = SearchResult(snippet)
            result.similarity_score = similarity_scores[i]
            result.calculate_final_score()
            
            results.append(result)
        
        return results
    
    fn get_architectural_patterns(inout self, pattern_type: String) -> List[SearchResult]:
        """
        Find architectural patterns across the codebase.
        
        Args:
            pattern_type: "middleware", "database", "api", "auth", etc.
            
        Returns:
            Code snippets matching the architectural pattern
        """
        # Define pattern-specific queries
        var pattern_query: String
        
        if pattern_type == "middleware":
            pattern_query = "middleware function request response next authentication"
        elif pattern_type == "database":
            pattern_query = "database connection pool query sql orm"
        elif pattern_type == "api":
            pattern_query = "api endpoint route handler http request response"
        elif pattern_type == "auth":
            pattern_query = "authentication authorization jwt token session"
        else:
            pattern_query = pattern_type
        
        return self.search(pattern_query, 15, "")
    
    fn update_search_context(inout self, current_project: String, current_file: String):
        """Update search context for better ranking."""
        self.search_context.current_project = current_project
        self.search_context.current_file = current_file
    
    fn _tokenize_code(self, code: String) -> Tensor[DType.float32]:
        """
        Tokenize code into embedding-ready format.
        
        TODO: Integrate with Tree-sitter for AST-based tokenization
        Currently uses simplified word-based tokenization.
        """
        # Simplified tokenization - in production would use Tree-sitter
        let max_tokens = 512
        var tokens = Tensor[DType.float32](max_tokens, MLAKernel.embed_dim)
        
        # Simple hash-based token encoding (placeholder)
        let words = code.split(" ")  # Simplified - would use proper tokenizer
        let num_tokens = min(len(words), max_tokens)
        
        for i in range(num_tokens):
            # Hash word to embedding space (simplified)
            let word_hash = hash(words[i]) % MLAKernel.embed_dim
            tokens[i, word_hash] = 1.0
        
        return tokens
    
    fn _tokenize_query(self, query: String) -> Tensor[DType.float32]:
        """Tokenize natural language query."""
        return self._tokenize_code(query)  # Reuse code tokenization for now
    
    fn _update_bmm_corpus(inout self):
        """Update BMM kernel with latest corpus embeddings."""
        let embeddings_matrix = self.code_corpus.embeddings_matrix
        self.bmm_kernel.load_corpus(embeddings_matrix)
    
    fn _apply_context_ranking(inout self, result: SearchResult):
        """Apply context-aware ranking to search result."""
        # Project relevance boost
        if result.snippet.project_name == self.search_context.current_project:
            result.project_relevance = 1.0
        else:
            result.project_relevance = 0.5
        
        # Recency boost (placeholder - would use git timestamps)
        result.recency_boost = 0.8  # Default recency
        
        # Context relevance (placeholder - would analyze surrounding code)
        result.context_relevance = 0.7  # Default context
    
    fn _sort_results_by_score(inout self, results: List[SearchResult]):
        """Sort search results by final score (descending)."""
        # Simple bubble sort - in production would use optimized sort
        let n = len(results)
        for i in range(n):
            for j in range(0, n - i - 1):
                if results[j].final_score < results[j + 1].final_score:
                    # Swap results
                    let temp = results[j]
                    results[j] = results[j + 1]
                    results[j + 1] = temp
    
    fn get_performance_report(self) -> String:
        """Get comprehensive performance report."""
        return (
            "=== Semantic Search Engine Performance ===\n" +
            self.mla_model.get_performance_stats() + "\n" +
            self.bmm_kernel.get_performance_metrics() + "\n" +
            self.performance_stats.get_report() + "\n" +
            "Total Indexed: " + str(self.total_embeddings) + " snippets"
        )

struct PerformanceTracker:
    """Track search performance metrics."""
    var total_searches: Int
    var total_search_time: Float64
    var average_results_per_search: Float32
    var current_search_start: Float64
    
    fn __init__(inout self):
        self.total_searches = 0
        self.total_search_time = 0.0
        self.average_results_per_search = 0.0
        self.current_search_start = 0.0
    
    fn start_search(inout self):
        """Mark start of search operation."""
        self.current_search_start = time.now().to_float64()
    
    fn end_search(inout self, num_results: Int):
        """Mark end of search and update metrics."""
        let search_time = time.now().to_float64() - self.current_search_start
        self.total_searches += 1
        self.total_search_time += search_time
        
        # Update rolling average
        let alpha: Float32 = 0.1  # Smoothing factor
        self.average_results_per_search = (
            (1.0 - alpha) * self.average_results_per_search +
            alpha * Float32(num_results)
        )
    
    fn get_report(self) -> String:
        """Generate performance report."""
        let avg_time = self.total_search_time / Float64(self.total_searches) if self.total_searches > 0 else 0.0
        
        return (
            "Performance Tracker:\n" +
            "- Total Searches: " + str(self.total_searches) + "\n" +
            "- Average Time: " + str(avg_time * 1000.0) + " ms\n" +
            "- Average Results: " + str(self.average_results_per_search)
        )

# High-level API functions
fn create_semantic_search_engine(corpus_size: Int = 50000) -> SemanticSearchEngine:
    """Create optimized semantic search engine."""
    return SemanticSearchEngine(corpus_size)

fn benchmark_full_search_pipeline(engine: SemanticSearchEngine, num_queries: Int = 50) -> String:
    """Benchmark complete search pipeline performance."""
    let test_queries = [
        "http client error handling",
        "database connection setup", 
        "authentication middleware",
        "async function implementation",
        "data validation patterns"
    ]
    
    let start_time = time.now()
    
    for i in range(num_queries):
        let query = test_queries[i % len(test_queries)]
        let _ = engine.search(query, 10)
    
    let end_time = time.now()
    let total_time = (end_time - start_time).to_float64()
    let avg_time = total_time / Float64(num_queries)
    
    return (
        "Full Pipeline Benchmark:\n" +
        "- Queries: " + str(num_queries) + "\n" +
        "- Total Time: " + str(total_time * 1000.0) + " ms\n" +
        "- Average per Query: " + str(avg_time * 1000.0) + " ms\n" +
        "- Queries per Second: " + str(1.0 / avg_time)
    )