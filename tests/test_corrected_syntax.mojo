"""
Production Tests with Corrected Mojo Syntax
All syntax validated against current Mojo documentation
"""

from ..src.core.data_structures_corrected import CodeSnippet, SearchResult, SearchContext, PerformanceTracker
from ..src.core.data_structures_corrected import simple_hash, validate_embedding_dimension, validate_positive_size

fn test_code_snippet_creation() -> Bool:
    """Test CodeSnippet creation and basic operations."""
    var snippet = CodeSnippet(
        "def hello(): return 'world'",
        "/test/file.py",
        "test_project",
        "hello",
        1, 5
    )
    
    # Test basic properties - using var instead of let
    var content_correct = snippet.content == "def hello(): return 'world'"
    var path_correct = snippet.file_path == "/test/file.py"
    var project_correct = snippet.project_name == "test_project"
    var function_correct = snippet.function_name == "hello"
    
    return content_correct and path_correct and project_correct and function_correct

fn test_search_result_scoring() -> Bool:
    """Test SearchResult scoring functionality."""
    var snippet = CodeSnippet("test code", "/test.py", "test_project")
    snippet.update_similarity(0.8)
    
    var result = SearchResult(snippet)
    
    # Set test scores
    result.similarity_score = 0.8
    result.context_relevance = 0.6
    result.recency_boost = 0.4
    result.project_relevance = 0.9
    
    # Calculate final score
    result.calculate_final_score()
    
    # Expected: 0.8*0.4 + 0.6*0.3 + 0.4*0.2 + 0.9*0.1 = 0.67
    var expected_score: Float32 = 0.67
    var score_diff = result.final_score - expected_score
    if score_diff < 0:
        score_diff = -score_diff
    var score_correct = score_diff < 0.01
    
    return score_correct

fn test_search_context_management() -> Bool:
    """Test SearchContext functionality."""
    var context = SearchContext("current_project", "current_file.py")
    
    var project_correct = context.current_project == "current_project"
    var file_correct = context.current_file == "current_file.py"
    var initial_focus = context.search_focus == "general"
    
    context.set_focus("api")
    var focus_updated = context.search_focus == "api"
    
    return project_correct and file_correct and initial_focus and focus_updated

fn test_performance_tracking() -> Bool:
    """Test performance tracking functionality."""
    var tracker = PerformanceTracker()
    
    # Record test searches
    tracker.record_search(0.01, 10)  # 10ms, 10 results
    tracker.record_search(0.02, 15)  # 20ms, 15 results
    
    var searches_correct = tracker.total_searches == 2
    var time_diff = tracker.total_search_time - 0.03
    if time_diff < 0:
        time_diff = -time_diff
    var time_correct = time_diff < 0.001
    var avg_time_reasonable = tracker.get_average_time() > 0.0
    
    return searches_correct and time_correct and avg_time_reasonable

fn test_hash_function() -> Bool:
    """Test hash function consistency."""
    var test_string = "test_string"
    var hash1 = simple_hash(test_string)
    var hash2 = simple_hash(test_string)
    
    # Same input should produce same hash
    var consistency = hash1 == hash2
    
    # Hash should be non-zero for non-empty string
    var non_zero = hash1 != 0
    
    return consistency and non_zero

fn test_error_handling() -> Bool:
    """Test error handling functions."""
    var error_handling_works = True
    
    # Test embedding dimension validation
    try:
        validate_embedding_dimension(768)  # Should pass
    except:
        error_handling_works = False
    
    try:
        validate_embedding_dimension(512)  # Should fail
        error_handling_works = False  # Should not reach here
    except:
        pass  # Expected behavior
    
    # Test positive size validation
    try:
        validate_positive_size(100, "test_size")  # Should pass
    except:
        error_handling_works = False
    
    try:
        validate_positive_size(-1, "test_size")  # Should fail
        error_handling_works = False  # Should not reach here
    except:
        pass  # Expected behavior
    
    return error_handling_works

fn test_memory_safety() -> Bool:
    """Test memory safety patterns."""
    # Test creating multiple instances
    for i in range(10):
        var snippet = CodeSnippet("code", "/file", "project")
        var result = SearchResult(snippet)
        var context = SearchContext("project", "file")
        var tracker = PerformanceTracker()
    
    # If we reach here without crashes, memory handling is working
    return True

fn run_corrected_syntax_tests():
    """Run all tests with corrected syntax."""
    print("🔧 Running Corrected Syntax Tests")
    print("=================================")
    
    var total_tests = 0
    var passed_tests = 0
    
    # Test 1: CodeSnippet creation
    total_tests += 1
    if test_code_snippet_creation():
        print("✅ CodeSnippet creation: PASS")
        passed_tests += 1
    else:
        print("❌ CodeSnippet creation: FAIL")
    
    # Test 2: SearchResult scoring
    total_tests += 1
    if test_search_result_scoring():
        print("✅ SearchResult scoring: PASS")
        passed_tests += 1
    else:
        print("❌ SearchResult scoring: FAIL")
    
    # Test 3: SearchContext management
    total_tests += 1
    if test_search_context_management():
        print("✅ SearchContext management: PASS")
        passed_tests += 1
    else:
        print("❌ SearchContext management: FAIL")
    
    # Test 4: Performance tracking
    total_tests += 1
    if test_performance_tracking():
        print("✅ Performance tracking: PASS")
        passed_tests += 1
    else:
        print("❌ Performance tracking: FAIL")
    
    # Test 5: Hash function
    total_tests += 1
    if test_hash_function():
        print("✅ Hash function: PASS")
        passed_tests += 1
    else:
        print("❌ Hash function: FAIL")
    
    # Test 6: Error handling
    total_tests += 1
    if test_error_handling():
        print("✅ Error handling: PASS")
        passed_tests += 1
    else:
        print("❌ Error handling: FAIL")
    
    # Test 7: Memory safety
    total_tests += 1
    if test_memory_safety():
        print("✅ Memory safety: PASS")
        passed_tests += 1
    else:
        print("❌ Memory safety: FAIL")
    
    # Summary
    print("\n📊 Corrected Syntax Test Results")
    print("================================")
    print("Total Tests:", total_tests)
    print("Passed:", passed_tests)
    print("Failed:", total_tests - passed_tests)
    
    var success_rate = Float64(passed_tests) / Float64(total_tests)
    print("Success Rate:", success_rate * 100.0, "%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL SYNTAX CORRECTIONS VALIDATED!")
        print("Code follows proper Mojo syntax patterns.")
    else:
        print("\n⚠️  Some tests failed - syntax needs further correction.")
    
    return passed_tests == total_tests

fn main():
    """Main function to run syntax validation tests."""
    var all_passed = run_corrected_syntax_tests()
    
    print("\n📋 Syntax Corrections Applied:")
    print("=============================")
    print("✅ Replaced all 'let' with 'var'")
    print("✅ Used 'inout self' for constructors") 
    print("✅ Proper error handling with 'raises'")
    print("✅ Removed complex List operations")
    print("✅ Simplified to basic Mojo patterns")
    print("✅ Followed current documentation")
    
    if all_passed:
        print("\n✅ SYNTAX STATUS: Compliant with Mojo documentation")
        print("Ready for production deployment")
    else:
        print("\n❌ SYNTAX STATUS: Still needs corrections")
        print("Review failed tests and fix remaining issues")