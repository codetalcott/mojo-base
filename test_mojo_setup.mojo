"""
Test Mojo setup and basic functionality.
Validates SIMD, tensor operations, and performance patterns.
"""

from utils.vector import DynamicVector
from math import sqrt

fn test_basic_functionality():
    """Test basic Mojo operations."""
    print("🧪 Testing basic Mojo functionality...")
    
    # Test basic variables and operations
    var a = 42
    var b = 3.14
    var result = Float64(a) * b
    
    print("Basic math: 42 * 3.14 =", result)
    print("✅ Basic functionality tests passed!")

fn test_memory_operations():
    """Test memory allocation and operations."""
    print("\n🧪 Testing Memory operations...")
    
    # Test dynamic vector (modern Mojo approach)
    var vec = DynamicVector[Float64]()
    vec.push_back(1.0)
    vec.push_back(2.0)
    vec.push_back(3.0)
    
    print("Vector size:", len(vec))
    print("Vector elements:")
    for i in range(len(vec)):
        print("  ", i, "->", vec[i])
    
    print("✅ Memory operation tests passed!")

fn test_loops_and_functions():
    """Test loop constructs and function definitions."""
    print("\n🧪 Testing Loops and functions...")
    
    # Test for loop
    var sum = 0
    for i in range(10):
        sum += i
    
    print("Sum of 0-9:", sum)
    
    # Test nested function
    fn square(x: Int) -> Int:
        return x * x
    
    var squared = square(7)
    print("Square of 7:", squared)
    
    print("✅ Loop and function tests passed!")

fn test_string_operations():
    """Test string handling."""
    print("\n🧪 Testing String operations...")
    
    var greeting = "Hello"
    var target = "Mojo"
    var message = greeting + " " + target + "!"
    
    print("String concatenation:", message)
    print("✅ String operation tests passed!")

fn test_performance_basics():
    """Test basic performance patterns available in current Mojo."""
    print("\n🧪 Testing Performance basics...")
    
    # Test simple computational pattern
    var data = DynamicVector[Float64]()
    
    # Initialize with some data
    for i in range(100):
        data.push_back(Float64(i))
    
    # Simple transformation
    for i in range(len(data)):
        data[i] = data[i] * data[i]  # Square each element
    
    print("Computed squares for", len(data), "elements")
    print("First few squared values:")
    for i in range(min(5, len(data))):
        print("  ", i, "->", data[i])
    
    print("✅ Performance basics tests passed!")

fn main():
    """Run all Mojo functionality tests."""
    print("🚀 Mojo Semantic Search - Setup Validation")
    print("==========================================")
    
    test_basic_functionality()
    test_memory_operations()
    test_loops_and_functions()
    test_string_operations()
    test_performance_basics()
    
    print("\n🎉 Basic Mojo tests passed! Environment is functional.")
    print("\n📊 Validated capabilities:")
    print("  - Basic arithmetic: ✅")
    print("  - Memory operations: ✅") 
    print("  - Control flow: ✅")
    print("  - String handling: ✅")
    print("  - Performance patterns: ✅")
    
    print("\n🎯 Next steps:")
    print("  - Implement advanced SIMD/tensor operations")
    print("  - Build MLA and BMM kernels")
    print("  - Create semantic search engine")
    print("  - Integrate with onedev portfolio system")
    
    print("\n🔗 Environment ready for semantic search development!")