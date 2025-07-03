"""
Test simple Mojo features that actually work
"""

from memory import UnsafePointer
from math import sqrt
from sys import simdwidthof
from random import random_float64

fn test_basic_operations():
    """Test basic operations that work."""
    print("🧪 Testing Basic Operations")
    print("===========================")
    
    # Test SIMD
    var width = simdwidthof[DType.float32]()
    print("✅ SIMD width for float32:", width)
    
    # Test UnsafePointer allocation and access
    var data = UnsafePointer[Float32].alloc(10)
    
    # Initialize some data
    for i in range(10):
        data[i] = Float32(i * 2.5)
    
    # Read back data
    print("✅ UnsafePointer data:", data[0], data[1], data[5])
    
    data.free()
    
    # Test math
    var sqrt_val = sqrt(16.0)
    print("✅ Math sqrt(16):", sqrt_val)
    
    # Test random
    var rand_val = random_float64(0.0, 1.0)
    print("✅ Random value:", rand_val)

fn basic_vector_operations():
    """Simple vector operations without complex APIs."""
    print("\n🧪 Testing Vector Operations")
    print("============================")
    
    var size = 100
    var vec_a = UnsafePointer[Float32].alloc(size)
    var vec_b = UnsafePointer[Float32].alloc(size)
    var result = UnsafePointer[Float32].alloc(size)
    
    # Initialize vectors
    for i in range(size):
        vec_a[i] = Float32(i)
        vec_b[i] = Float32(i * 2)
    
    # Simple dot product computation
    var dot_product: Float32 = 0.0
    for i in range(size):
        dot_product += vec_a[i] * vec_b[i]
    
    # Vector addition
    for i in range(size):
        result[i] = vec_a[i] + vec_b[i]
    
    print("✅ Dot product of first 100 integers:", dot_product)
    print("✅ Vector addition result[5]:", result[5])
    
    vec_a.free()
    vec_b.free()
    result.free()

fn similarity_computation_demo():
    """Demo of similarity computation using available features."""
    print("\n🧪 Testing Similarity Computation")
    print("==================================")
    
    var embed_dim = 128
    var query = UnsafePointer[Float32].alloc(embed_dim)
    var corpus_item = UnsafePointer[Float32].alloc(embed_dim)
    
    # Initialize with some test data
    for i in range(embed_dim):
        query[i] = Float32(i) / Float32(embed_dim)
        corpus_item[i] = Float32(i + 10) / Float32(embed_dim)
    
    # Compute cosine similarity
    var dot_product: Float32 = 0.0
    var query_norm: Float32 = 0.0
    var corpus_norm: Float32 = 0.0
    
    for i in range(embed_dim):
        dot_product += query[i] * corpus_item[i]
        query_norm += query[i] * query[i]
        corpus_norm += corpus_item[i] * corpus_item[i]
    
    query_norm = sqrt(query_norm)
    corpus_norm = sqrt(corpus_norm)
    
    var similarity = dot_product / (query_norm * corpus_norm)
    
    print("✅ Cosine similarity:", similarity)
    
    query.free()
    corpus_item.free()

fn main():
    """Test simple features that work."""
    print("🔍 Testing Available Mojo Features (Simple)")
    print("===========================================")
    
    test_basic_operations()
    basic_vector_operations()
    similarity_computation_demo()
    
    print("\n📋 Status Summary:")
    print("✅ UnsafePointer: WORKING")
    print("✅ Basic math: WORKING") 
    print("✅ Memory allocation: WORKING")
    print("✅ For loops: WORKING")
    print("✅ Vector operations: MANUAL IMPLEMENTATION WORKS")
    
    print("\n💡 Strategy: Build kernels using these confirmed features")