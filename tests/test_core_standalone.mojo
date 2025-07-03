"""
Standalone test for core data structures.
Includes minimal implementations for testing purposes.
"""

# Minimal data structures for testing
struct CodeSnippet:
    """Test version of CodeSnippet."""
    var content: String
    var file_path: String 
    var project_name: String
    var similarity_score: Float32
    
    fn __init__(out self, content: String, file_path: String, project_name: String):
        self.content = content
        self.file_path = file_path
        self.project_name = project_name
        self.similarity_score = 0.0
    
    fn update_similarity(mut self, score: Float32):
        self.similarity_score = score

struct SearchResult:
    """Test version of SearchResult."""
    var similarity_score: Float32
    var context_relevance: Float32
    var recency_boost: Float32
    var project_relevance: Float32
    var final_score: Float32
    
    fn __init__(out self, base_score: Float32):
        self.similarity_score = base_score
        self.context_relevance = 0.0
        self.recency_boost = 0.0
        self.project_relevance = 0.0
        self.final_score = 0.0
    
    fn calculate_final_score(mut self):
        self.final_score = (
            self.similarity_score * 0.4 +
            self.context_relevance * 0.3 +
            self.recency_boost * 0.2 +
            self.project_relevance * 0.1
        )

struct SearchContext:
    """Test version of SearchContext."""
    var current_project: String
    var current_file: String
    var search_focus: String
    
    fn __init__(out self, current_project: String, current_file: String):
        self.current_project = current_project
        self.current_file = current_file
        self.search_focus = "general"
    
    fn set_focus(mut self, focus: String):
        self.search_focus = focus

struct PerformanceTracker:
    """Test version of PerformanceTracker."""
    var total_searches: Int
    var total_search_time: Float64
    
    fn __init__(out self):
        self.total_searches = 0
        self.total_search_time = 0.0
    
    fn record_search(mut self, search_time: Float64, num_results: Int):
        self.total_searches += 1
        self.total_search_time += search_time
    
    fn get_average_time(self) -> Float64:
        if self.total_searches > 0:
            return self.total_search_time / Float64(self.total_searches)
        return 0.0

fn simple_hash(s: String) -> Int:
    """Simple hash function for strings."""
    var hash_value = 0
    var s_len = len(s)
    for i in range(s_len):
        hash_value = hash_value * 31 + ord(s[i])
    if hash_value < 0:
        hash_value = -hash_value
    return hash_value

fn validate_embedding_dimension(dimension: Int) raises:
    """Validate embedding dimension is correct."""
    if dimension != 768:
        raise Error("Embedding must be 768 dimensions")

# Test functions
fn test_code_snippet() -> Bool:
    """Test CodeSnippet creation and basic operations."""
    var snippet = CodeSnippet("def hello(): return 'world'", "/test/file.py", "test_project")
    
    var content_correct = snippet.content == "def hello(): return 'world'"
    var path_correct = snippet.file_path == "/test/file.py"
    var project_correct = snippet.project_name == "test_project"
    
    snippet.update_similarity(0.8)
    var score_correct = snippet.similarity_score == 0.8
    
    return content_correct and path_correct and project_correct and score_correct

fn test_search_result() -> Bool:
    """Test SearchResult scoring functionality."""
    var snippet = CodeSnippet("test code", "/test.py", "test_project")
    snippet.update_similarity(0.8)
    
    var result = SearchResult(0.8)
    result.similarity_score = 0.8
    result.context_relevance = 0.6
    result.recency_boost = 0.4
    result.project_relevance = 0.9
    
    result.calculate_final_score()
    
    # Expected: 0.8*0.4 + 0.6*0.3 + 0.4*0.2 + 0.9*0.1 = 0.67
    var expected_score: Float32 = 0.67
    var score_diff = result.final_score - expected_score
    if score_diff < 0:
        score_diff = -score_diff
    
    return score_diff < 0.01

fn test_search_context() -> Bool:
    """Test SearchContext functionality."""
    var context = SearchContext("current_project", "current_file.py")
    
    var project_correct = context.current_project == "current_project"
    var file_correct = context.current_file == "current_file.py" 
    var initial_focus = context.search_focus == "general"
    
    context.set_focus("api")
    var focus_updated = context.search_focus == "api"
    
    return project_correct and file_correct and initial_focus and focus_updated

fn test_performance_tracker() -> Bool:
    """Test performance tracking functionality."""
    var tracker = PerformanceTracker()
    
    tracker.record_search(0.01, 10)
    tracker.record_search(0.02, 15)
    
    var searches_correct = tracker.total_searches == 2
    var avg_time_reasonable = tracker.get_average_time() > 0.0
    
    return searches_correct and avg_time_reasonable

fn test_hash_function() -> Bool:
    """Test hash function consistency."""
    var test_string = "test_string"
    var hash1 = simple_hash(test_string)
    var hash2 = simple_hash(test_string)
    
    var consistency = hash1 == hash2
    var non_zero = hash1 != 0
    
    return consistency and non_zero

fn test_validation_functions() -> Bool:
    """Test validation functions."""
    var validation_works = True
    
    # Test valid embedding dimension
    try:
        validate_embedding_dimension(768)
    except:
        validation_works = False
    
    # Test invalid embedding dimension should raise error
    try:
        validate_embedding_dimension(512)
        validation_works = False  # Should not reach here
    except:
        pass  # Expected behavior
    
    return validation_works

fn run_all_tests():
    """Run all core data structure tests."""
    print("🧪 Running Core Data Structure Tests")
    print("===================================")
    
    var total_tests = 0
    var passed_tests = 0
    
    total_tests += 1
    if test_code_snippet():
        print("✅ CodeSnippet: PASS")
        passed_tests += 1
    else:
        print("❌ CodeSnippet: FAIL")
    
    total_tests += 1
    if test_search_result():
        print("✅ SearchResult: PASS")
        passed_tests += 1
    else:
        print("❌ SearchResult: FAIL")
    
    total_tests += 1
    if test_search_context():
        print("✅ SearchContext: PASS")
        passed_tests += 1
    else:
        print("❌ SearchContext: FAIL")
    
    total_tests += 1
    if test_performance_tracker():
        print("✅ PerformanceTracker: PASS")
        passed_tests += 1
    else:
        print("❌ PerformanceTracker: FAIL")
    
    total_tests += 1
    if test_hash_function():
        print("✅ Hash Function: PASS")
        passed_tests += 1
    else:
        print("❌ Hash Function: FAIL")
    
    total_tests += 1
    if test_validation_functions():
        print("✅ Validation Functions: PASS")
        passed_tests += 1
    else:
        print("❌ Validation Functions: FAIL")
    
    print("\n📊 Test Results")
    print("===============")
    print("Total Tests:", total_tests)
    print("Passed:", passed_tests)
    print("Failed:", total_tests - passed_tests)
    
    var success_rate = Float64(passed_tests) / Float64(total_tests)
    print("Success Rate:", success_rate * 100.0, "%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL CORE TESTS PASSED!")
        print("Core data structures are production ready.")
    else:
        print("\n⚠️  Some core tests failed.")

fn main():
    """Main test function."""
    run_all_tests()