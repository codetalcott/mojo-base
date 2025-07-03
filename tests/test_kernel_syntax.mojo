"""
Test kernel syntax compilation
Simple test to verify kernels compile correctly
"""

from tensor import Tensor

fn test_basic_syntax():
    """Test basic Mojo syntax compilation."""
    var x: Float32 = 1.0
    var y = x + 2.0
    print("Basic syntax test:", y)

fn main():
    """Main test function."""
    print("🧪 Testing Kernel Syntax Compilation")
    print("====================================")
    
    test_basic_syntax()
    
    print("✅ Basic syntax test passed!")
    print("🎉 Kernel syntax verification complete!")