"""
Test what features are actually available in this Mojo installation
"""

# Test basic imports that we know work
from algorithm import parallelize, vectorize
from memory import UnsafePointer
from math import sqrt, exp
from sys import simdwidthof
from random import random_float64

# Test what doesn't work yet but we need
# from buffer.buffer import NDBuffer  # Test if available
# from gpu import WARP_SIZE  # Test if available

fn test_basic_features():
    """Test basic features that should work."""
    print("🧪 Testing Basic Mojo Features")
    print("=============================")
    
    # Test SIMD
    var width = simdwidthof[DType.float32]()
    print("✅ SIMD width:", width)
    
    # Test UnsafePointer
    var ptr = UnsafePointer[Float32].alloc(10)
    ptr[0] = 1.5
    var value = ptr[0]
    ptr.free()
    print("✅ UnsafePointer:", value)
    
    # Test math functions
    var sqrt_val = sqrt(4.0)
    print("✅ Math functions:", sqrt_val)
    
    # Test random
    var rand_val = random_float64(0.0, 1.0)
    print("✅ Random:", rand_val)

fn test_vectorization():
    """Test if vectorize function works."""
    print("\n🧪 Testing Vectorization")
    print("========================")
    
    var data = UnsafePointer[Float32].alloc(100)
    
    @parameter
    fn init_data(i: Int):
        data[i] = Float32(i)
    
    vectorize[4, init_data](100)
    
    print("✅ Vectorization works, first values:", data[0], data[1], data[2])
    data.free()

fn test_parallelization():
    """Test if parallelize function works."""
    print("\n🧪 Testing Parallelization")
    print("==========================")
    
    var results = UnsafePointer[Float32].alloc(10)
    
    @parameter
    fn compute_square(i: Int):
        results[i] = Float32(i * i)
    
    parallelize[compute_square](10)
    
    print("✅ Parallelize works, squares:", results[0], results[1], results[4], results[9])
    results.free()

fn main():
    """Test available features."""
    print("🔍 Discovering Available Mojo Features")
    print("======================================")
    
    test_basic_features()
    test_vectorization()  
    test_parallelization()
    
    print("\n📋 Summary:")
    print("✅ Basic types and math: AVAILABLE")
    print("✅ UnsafePointer: AVAILABLE") 
    print("✅ SIMD operations: AVAILABLE")
    print("✅ Vectorize/Parallelize: AVAILABLE")
    print("❓ NDBuffer/GPU APIs: NEED TESTING")
    
    print("\n💡 Recommendation: Build kernels using confirmed available features")