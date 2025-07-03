"""
Unit Tests for Core Data Structures
TDD approach to ensure production reliability and correctness
"""

from testing import assert_equal, assert_true, assert_false
from tensor import Tensor
from utils.list import List
from DType import DType
from ..src.core.data_structures import CodeSnippet, SearchResult, SearchContext, EmbeddingCache, CodeCorpus

# Test Result tracking
struct TestResults:
    var total_tests: Int
    var passed_tests: Int
    var failed_tests: Int
    
    fn __init__(inout self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    fn record_test(inout self, passed: Bool, test_name: String):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print("✅ PASS:", test_name)
        else:
            self.failed_tests += 1
            print("❌ FAIL:", test_name)
    
    fn print_summary(self):
        print("\n📊 Test Summary:")
        print("================")
        print("Total Tests:", self.total_tests)
        print("Passed:", self.passed_tests)
        print("Failed:", self.failed_tests)
        print("Success Rate:", Float64(self.passed_tests) / Float64(self.total_tests) * 100.0, "%")

# Test Suite for CodeSnippet
fn test_code_snippet_creation() -> Bool:
    """Test CodeSnippet creation and basic operations."""
    try:
        var snippet = CodeSnippet(
            "def hello(): return 'world'",
            "/test/file.py",
            "test_project",
            "hello",
            1, 5
        )
        
        # Test basic properties
        let content_correct = snippet.content == "def hello(): return 'world'"
        let path_correct = snippet.file_path == "/test/file.py"
        let project_correct = snippet.project_name == "test_project"
        let function_correct = snippet.function_name == "hello"
        let line_start_correct = snippet.line_start == 1
        let line_end_correct = snippet.line_end == 5
        
        return content_correct and path_correct and project_correct and function_correct and line_start_correct and line_end_correct
    except:
        return False

fn test_code_snippet_dependencies() -> Bool:
    """Test dependency management in CodeSnippet."""
    try:
        var snippet = CodeSnippet("code", "/path", "project")
        
        # Test adding dependencies
        snippet.add_dependency("numpy")
        snippet.add_dependency("pandas")
        
        # Verify dependencies were added
        let has_dependencies = len(snippet.dependencies) == 2
        return has_dependencies
    except:
        return False

fn test_code_snippet_embedding() -> Bool:
    """Test embedding operations in CodeSnippet."""
    try:
        var snippet = CodeSnippet("code", "/path", "project")
        var test_embedding = Tensor[DType.float32](768)
        
        # Initialize test embedding
        for i in range(768):
            test_embedding[i] = Float32(i) / 768.0
        
        # Test setting embedding
        snippet.set_embedding(test_embedding)
        
        # Verify embedding was set correctly
        let embedding_correct = snippet.embedding[0] == 0.0 and snippet.embedding[767] == Float32(767) / 768.0
        return embedding_correct
    except:
        return False

# Test Suite for SearchResult
fn test_search_result_creation() -> Bool:
    """Test SearchResult creation and score calculation."""
    try:
        var snippet = CodeSnippet("test code", "/test.py", "test_project")
        snippet.update_similarity(0.8)
        
        var result = SearchResult(snippet)
        
        # Test initial state
        let similarity_correct = result.similarity_score == 0.8
        let initial_scores_zero = (result.context_relevance == 0.0 and 
                                 result.recency_boost == 0.0 and 
                                 result.project_relevance == 0.0 and 
                                 result.final_score == 0.0)
        
        return similarity_correct and initial_scores_zero
    except:
        return False

fn test_search_result_final_score() -> Bool:
    """Test final score calculation with weighted combination."""
    try:
        var snippet = CodeSnippet("test code", "/test.py", "test_project")
        var result = SearchResult(snippet)
        
        # Set test scores
        result.similarity_score = 0.8
        result.context_relevance = 0.6
        result.recency_boost = 0.4
        result.project_relevance = 0.9
        
        # Calculate final score
        result.calculate_final_score()
        
        # Expected: 0.8*0.4 + 0.6*0.3 + 0.4*0.2 + 0.9*0.1 = 0.32 + 0.18 + 0.08 + 0.09 = 0.67
        let expected_score = 0.67
        let score_correct = abs(result.final_score - expected_score) < 0.001
        
        return score_correct
    except:
        return False

# Test Suite for SearchContext
fn test_search_context_creation() -> Bool:
    """Test SearchContext initialization and basic operations."""
    try:
        var context = SearchContext("current_project", "current_file.py")
        
        let project_correct = context.current_project == "current_project"
        let file_correct = context.current_file == "current_file.py"
        let initial_state = (len(context.recent_queries) == 0 and 
                            len(context.preferred_languages) == 0 and 
                            context.search_focus == "general")
        
        return project_correct and file_correct and initial_state
    except:
        return False

fn test_search_context_query_tracking() -> Bool:
    """Test recent query tracking functionality."""
    try:
        var context = SearchContext()
        
        # Add test queries
        context.add_recent_query("test query 1")
        context.add_recent_query("test query 2")
        context.add_recent_query("test query 3")
        
        let query_count_correct = len(context.recent_queries) == 3
        return query_count_correct
    except:
        return False

fn test_search_context_focus_setting() -> Bool:
    """Test search focus setting functionality."""
    try:
        var context = SearchContext()
        
        context.set_focus("api")
        let focus_api = context.search_focus == "api"
        
        context.set_focus("patterns")
        let focus_patterns = context.search_focus == "patterns"
        
        return focus_api and focus_patterns
    except:
        return False

# Test Suite for EmbeddingCache
fn test_embedding_cache_creation() -> Bool:
    """Test EmbeddingCache initialization."""
    try:
        var cache = EmbeddingCache(1000)
        
        let size_correct = cache.max_size == 1000
        let initial_state = (cache.cache_size == 0 and 
                            cache.hit_count == 0 and 
                            cache.miss_count == 0)
        
        return size_correct and initial_state
    except:
        return False

fn test_embedding_cache_efficiency() -> Bool:
    """Test cache efficiency calculation."""
    try:
        var cache = EmbeddingCache(100)
        
        # Simulate cache hits and misses
        cache.hit_count = 80
        cache.miss_count = 20
        
        let efficiency = cache.get_cache_efficiency()
        let expected_efficiency = 0.8  # 80/(80+20) = 0.8
        let efficiency_correct = abs(efficiency - expected_efficiency) < 0.001
        
        return efficiency_correct
    except:
        return False

# Test Suite for CodeCorpus
fn test_code_corpus_creation() -> Bool:
    """Test CodeCorpus initialization."""
    try:
        var corpus = CodeCorpus(5000)
        
        let initial_state = (corpus.total_snippets == 0 and 
                            len(corpus.snippets) == 0 and 
                            len(corpus.project_index) == 0 and 
                            len(corpus.file_index) == 0)
        
        return initial_state
    except:
        return False

fn test_code_corpus_add_snippet() -> Bool:
    """Test adding snippets to corpus."""
    try:
        var corpus = CodeCorpus(100)
        var snippet1 = CodeSnippet("code1", "/file1.py", "project1")
        var snippet2 = CodeSnippet("code2", "/file2.py", "project2")
        
        corpus.add_snippet(snippet1)
        corpus.add_snippet(snippet2)
        
        let count_correct = corpus.total_snippets == 2
        let snippets_added = len(corpus.snippets) == 2
        let indices_updated = (len(corpus.project_index) == 2 and 
                              len(corpus.file_index) == 2)
        
        return count_correct and snippets_added and indices_updated
    except:
        return False

fn test_code_corpus_project_filtering() -> Bool:
    """Test project-based filtering functionality."""
    try:
        var corpus = CodeCorpus(100)
        
        # Add snippets from different projects
        var snippet1 = CodeSnippet("code1", "/file1.py", "project_a")
        var snippet2 = CodeSnippet("code2", "/file2.py", "project_b")
        var snippet3 = CodeSnippet("code3", "/file3.py", "project_a")
        
        corpus.add_snippet(snippet1)
        corpus.add_snippet(snippet2)
        corpus.add_snippet(snippet3)
        
        # Filter by project_a
        let project_a_indices = corpus.filter_by_project("project_a")
        let project_a_count = len(project_a_indices) == 2
        
        # Filter by project_b
        let project_b_indices = corpus.filter_by_project("project_b")
        let project_b_count = len(project_b_indices) == 1
        
        return project_a_count and project_b_count
    except:
        return False

# Error handling and edge case tests
fn test_error_handling_empty_inputs() -> Bool:
    """Test error handling for empty inputs."""
    try:
        # Test empty CodeSnippet
        var empty_snippet = CodeSnippet("", "", "")
        let empty_content = empty_snippet.content == ""
        
        # Test empty SearchContext
        var empty_context = SearchContext("", "")
        let empty_project = empty_context.current_project == ""
        
        return empty_content and empty_project
    except:
        return False

fn test_memory_safety() -> Bool:
    """Test memory safety for large allocations."""
    try:
        # Test large embedding cache
        var large_cache = EmbeddingCache(10000)
        let cache_created = large_cache.max_size == 10000
        
        # Test large corpus
        var large_corpus = CodeCorpus(50000)
        let corpus_created = large_corpus.total_snippets == 0
        
        return cache_created and corpus_created
    except:
        return False

# Main test runner
fn run_all_tests():
    """Run comprehensive test suite for core data structures."""
    print("🧪 Running Core Data Structures Test Suite")
    print("==========================================")
    
    var results = TestResults()
    
    # CodeSnippet tests
    print("\n📝 Testing CodeSnippet...")
    results.record_test(test_code_snippet_creation(), "CodeSnippet Creation")
    results.record_test(test_code_snippet_dependencies(), "CodeSnippet Dependencies")
    results.record_test(test_code_snippet_embedding(), "CodeSnippet Embedding")
    
    # SearchResult tests
    print("\n🔍 Testing SearchResult...")
    results.record_test(test_search_result_creation(), "SearchResult Creation")
    results.record_test(test_search_result_final_score(), "SearchResult Final Score")
    
    # SearchContext tests
    print("\n🎯 Testing SearchContext...")
    results.record_test(test_search_context_creation(), "SearchContext Creation")
    results.record_test(test_search_context_query_tracking(), "SearchContext Query Tracking")
    results.record_test(test_search_context_focus_setting(), "SearchContext Focus Setting")
    
    # EmbeddingCache tests
    print("\n💾 Testing EmbeddingCache...")
    results.record_test(test_embedding_cache_creation(), "EmbeddingCache Creation")
    results.record_test(test_embedding_cache_efficiency(), "EmbeddingCache Efficiency")
    
    # CodeCorpus tests
    print("\n📚 Testing CodeCorpus...")
    results.record_test(test_code_corpus_creation(), "CodeCorpus Creation")
    results.record_test(test_code_corpus_add_snippet(), "CodeCorpus Add Snippet")
    results.record_test(test_code_corpus_project_filtering(), "CodeCorpus Project Filtering")
    
    # Error handling tests
    print("\n⚠️  Testing Error Handling...")
    results.record_test(test_error_handling_empty_inputs(), "Empty Inputs Handling")
    results.record_test(test_memory_safety(), "Memory Safety")
    
    # Print final results
    results.print_summary()
    
    if results.failed_tests == 0:
        print("\n🎉 All tests passed! Core data structures are ready for production.")
    else:
        print("\n⚠️  Some tests failed. Issues must be addressed before production.")

fn main():
    """Main function to run all tests."""
    run_all_tests()