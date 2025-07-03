"""
Simple kernel syntax test
Tests basic Mojo syntax without complex imports
"""

struct SimpleKernel:
    """Simple test kernel for syntax verification."""
    var size: Int
    var data: Float32
    
    fn __init__(out self, size: Int):
        """Initialize simple kernel."""
        self.size = size
        self.data = 1.0
    
    fn process(mut self, value: Float32) -> Float32:
        """Simple processing function."""
        self.data = self.data * value
        return self.data
    
    fn get_size(self) -> Int:
        """Get kernel size."""
        return self.size

fn test_kernel_syntax() -> Bool:
    """Test kernel syntax compilation."""
    var kernel = SimpleKernel(10)
    var result = kernel.process(2.0)
    var size = kernel.get_size()
    
    return result == 2.0 and size == 10

fn main():
    """Main test function."""
    print("🧪 Testing Simple Kernel Syntax")
    print("===============================")
    
    if test_kernel_syntax():
        print("✅ Kernel syntax test passed!")
        print("✅ struct definitions work correctly")
        print("✅ fn __init__(out self, ...) syntax correct")
        print("✅ fn method(mut self, ...) syntax correct")
        print("✅ fn method(self) -> Type syntax correct")
    else:
        print("❌ Kernel syntax test failed!")
    
    print("\n🎉 Kernel syntax verification complete!")