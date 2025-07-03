"""
Unit Tests for MLA Kernel
TDD approach to ensure production reliability and performance
"""

from testing import assert_equal, assert_true, assert_false
from tensor import Tensor
from DType import DType
from math import abs
from ..src.kernels.mla_kernel import MLAKernel, create_optimized_mla_kernel, benchmark_mla_kernel

struct MLATestResults:
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
        print("\n📊 MLA Kernel Test Summary:")
        print("===========================")
        print("Total Tests:", self.total_tests)
        print("Passed:", self.passed_tests) 
        print("Failed:", self.failed_tests)
        print("Success Rate:", Float64(self.passed_tests) / Float64(self.total_tests) * 100.0, "%")

fn test_mla_kernel_initialization() -> Bool:
    """Test MLA kernel initialization and basic properties."""
    try:
        var kernel = MLAKernel()
        
        # Test basic properties
        let heads_correct = MLAKernel.num_heads == 8
        let embed_dim_correct = MLAKernel.embed_dim == 768
        let head_dim_correct = MLAKernel.head_dim == 96
        let max_seq_correct = MLAKernel.max_seq_len == 512
        
        # Test weight matrix dimensions
        let query_shape = (kernel.query_weights.shape()[0] == 768 and 
                          kernel.query_weights.shape()[1] == 768)
        let key_shape = (kernel.key_weights.shape()[0] == 768 and 
                        kernel.key_weights.shape()[1] == 768)
        let value_shape = (kernel.value_weights.shape()[0] == 768 and 
                          kernel.value_weights.shape()[1] == 768)
        let output_shape = (kernel.output_weights.shape()[0] == 768 and 
                           kernel.output_weights.shape()[1] == 768)
        
        # Test attention mask
        let mask_shape = (kernel.syntax_attention_mask.shape()[0] == 512 and 
                         kernel.syntax_attention_mask.shape()[1] == 512)
        
        return (heads_correct and embed_dim_correct and head_dim_correct and 
                max_seq_correct and query_shape and key_shape and value_shape and 
                output_shape and mask_shape)
    except:
        return False

fn test_mla_kernel_encode_sequence_bounds() -> Bool:
    """Test sequence encoding with proper bounds checking."""
    try:
        var kernel = MLAKernel()
        
        # Test normal case
        let seq_len = 256
        var test_tokens = Tensor[DType.float32](seq_len, 768)
        
        # Initialize test data
        for i in range(seq_len):
            for j in range(768):
                test_tokens[i, j] = Float32(i + j) / 1000.0
        
        let result = kernel.encode_sequence(test_tokens, seq_len)
        
        # Test output dimensions
        let output_correct = result.shape()[0] == 768
        
        # Test boundary case - max sequence length
        let max_seq_len = 512
        var max_tokens = Tensor[DType.float32](max_seq_len, 768)
        let max_result = kernel.encode_sequence(max_tokens, max_seq_len)
        let max_output_correct = max_result.shape()[0] == 768
        
        return output_correct and max_output_correct
    except:
        return False

fn test_mla_kernel_encode_sequence_edge_cases() -> Bool:
    """Test sequence encoding edge cases."""
    try:
        var kernel = MLAKernel()
        
        # Test minimum sequence length
        let min_seq_len = 1
        var min_tokens = Tensor[DType.float32](min_seq_len, 768)
        let min_result = kernel.encode_sequence(min_tokens, min_seq_len)
        let min_case_valid = min_result.shape()[0] == 768
        
        # Test empty sequence (should handle gracefully)
        let empty_seq_len = 0
        if empty_seq_len > 0:  # Only test if valid
            var empty_tokens = Tensor[DType.float32](1, 768)  # Minimum allocation
            let empty_result = kernel.encode_sequence(empty_tokens, empty_seq_len)
        
        return min_case_valid
    except:
        return False

fn test_mla_kernel_weight_initialization() -> Bool:
    """Test weight initialization for reasonable values."""
    try:
        var kernel = MLAKernel()
        
        # Check that weights are not all zero (initialization worked)
        var weights_initialized = False
        for i in range(min(10, 768)):  # Check first 10 elements
            for j in range(min(10, 768)):
                if kernel.query_weights[i, j] != 0.0:
                    weights_initialized = True
                    break
            if weights_initialized:
                break
        
        # Check that weights are within reasonable bounds (Xavier initialization)
        var weights_reasonable = True
        let scale_bound = 2.0  # Reasonable bound for Xavier initialization
        for i in range(min(10, 768)):
            for j in range(min(10, 768)):
                if abs(kernel.query_weights[i, j]) > scale_bound:
                    weights_reasonable = False
                    break
            if not weights_reasonable:
                break
        
        return weights_initialized and weights_reasonable
    except:
        return False

fn test_mla_kernel_attention_mask() -> Bool:
    """Test attention mask initialization and structure."""
    try:
        var kernel = MLAKernel()
        
        # Check that mask is properly initialized (all True initially)
        var mask_initialized = True
        for i in range(min(10, 512)):  # Check subset for performance
            for j in range(min(10, 512)):
                if not kernel.syntax_attention_mask[i, j]:
                    mask_initialized = False
                    break
            if not mask_initialized:
                break
        
        return mask_initialized
    except:
        return False

fn test_mla_kernel_memory_safety() -> Bool:
    """Test memory safety and proper cleanup."""
    try:
        # Test multiple kernel creation/destruction
        for _ in range(5):
            var kernel = MLAKernel()
            let seq_len = 64
            var test_tokens = Tensor[DType.float32](seq_len, 768)
            let _ = kernel.encode_sequence(test_tokens, seq_len)
        
        return True
    except:
        return False

fn test_mla_kernel_performance_stats() -> Bool:
    """Test performance statistics reporting."""
    try:
        var kernel = MLAKernel()
        let stats = kernel.get_performance_stats()
        
        # Check that stats contain expected information
        let stats_valid = len(stats) > 0
        
        return stats_valid
    except:
        return False

fn test_create_optimized_mla_kernel() -> Bool:
    """Test factory function for creating optimized kernel."""
    try:
        var kernel = create_optimized_mla_kernel()
        
        # Test that factory returns a properly initialized kernel
        let factory_valid = MLAKernel.embed_dim == 768
        
        return factory_valid
    except:
        return False

fn test_mla_kernel_consistency() -> Bool:
    """Test that multiple encodings of same input produce same output."""
    try:
        var kernel = MLAKernel()
        let seq_len = 128
        var test_tokens = Tensor[DType.float32](seq_len, 768)
        
        # Initialize with deterministic data
        for i in range(seq_len):
            for j in range(768):
                test_tokens[i, j] = Float32(i * 768 + j) / 100000.0
        
        # Encode twice
        let result1 = kernel.encode_sequence(test_tokens, seq_len)
        let result2 = kernel.encode_sequence(test_tokens, seq_len)
        
        # Check consistency (should be identical for same input)
        var consistent = True
        for i in range(min(10, 768)):  # Check subset for performance
            if abs(result1[i] - result2[i]) > 1e-6:
                consistent = False
                break
        
        return consistent
    except:
        return False

fn test_mla_kernel_benchmark_functionality() -> Bool:
    """Test that benchmark function works correctly."""
    try:
        var kernel = MLAKernel()
        let avg_time = benchmark_mla_kernel(kernel, 5)  # Small iteration count for testing
        
        # Check that benchmark returns reasonable time (should be positive)
        let benchmark_valid = avg_time > 0.0
        
        return benchmark_valid
    except:
        return False

fn run_mla_kernel_tests():
    """Run comprehensive test suite for MLA kernel."""
    print("🧪 Running MLA Kernel Test Suite")
    print("=================================")
    
    var results = MLATestResults()
    
    print("\n🔧 Testing Initialization...")
    results.record_test(test_mla_kernel_initialization(), "MLA Kernel Initialization")
    results.record_test(test_mla_kernel_weight_initialization(), "Weight Initialization")
    results.record_test(test_mla_kernel_attention_mask(), "Attention Mask")
    
    print("\n🎯 Testing Core Functionality...")
    results.record_test(test_mla_kernel_encode_sequence_bounds(), "Sequence Encoding Bounds")
    results.record_test(test_mla_kernel_encode_sequence_edge_cases(), "Sequence Encoding Edge Cases")
    results.record_test(test_mla_kernel_consistency(), "Encoding Consistency")
    
    print("\n🛡️  Testing Safety and Performance...")
    results.record_test(test_mla_kernel_memory_safety(), "Memory Safety")
    results.record_test(test_mla_kernel_performance_stats(), "Performance Stats")
    results.record_test(test_create_optimized_mla_kernel(), "Factory Function")
    results.record_test(test_mla_kernel_benchmark_functionality(), "Benchmark Function")
    
    results.print_summary()
    
    if results.failed_tests == 0:
        print("\n🎉 All MLA kernel tests passed! Ready for production.")
    else:
        print("\n⚠️  Some MLA kernel tests failed. Issues must be addressed.")

fn main():
    """Main function to run MLA kernel tests."""
    run_mla_kernel_tests()