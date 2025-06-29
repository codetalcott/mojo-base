"""
Simple Mojo test to validate environment setup.
"""

fn test_basic_operations():
    """Test basic Mojo operations."""
    print("🧪 Testing basic Mojo functionality...")
    
    # Basic variables and arithmetic
    var a = 42
    var b = 3.14
    var result = Float64(a) * b
    
    print("Basic math: 42 * 3.14 =", result)
    
    # Test loops
    var sum = 0
    for i in range(10):
        sum += i
    
    print("Sum of 0-9:", sum)
    
    # Test function definition
    fn square(x: Int) -> Int:
        return x * x
    
    var squared = square(7)
    print("Square of 7:", squared)
    
    print("✅ Basic functionality tests passed!")

fn test_string_operations():
    """Test string handling."""
    print("\n🧪 Testing String operations...")
    
    var greeting = "Hello"
    var target = "Mojo"
    var message = greeting + " " + target + "!"
    
    print("String concatenation:", message)
    print("✅ String tests passed!")

fn main():
    """Main test function."""
    print("🚀 Mojo Environment Validation")
    print("==============================")
    
    test_basic_operations()
    test_string_operations()
    
    print("\n🎉 Mojo environment is working!")
    print("\n📊 Validated:")
    print("  - Basic arithmetic: ✅")
    print("  - Control flow: ✅")
    print("  - Functions: ✅")
    print("  - String operations: ✅")
    
    print("\n🎯 Ready for semantic search implementation!")