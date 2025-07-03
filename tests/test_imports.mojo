"""
Test available imports in current Mojo installation
"""

# Test basic imports
from algorithm import parallelize, vectorize
from math import sqrt, exp
from memory import UnsafePointer
from sys import simdwidthof

fn test_basic_imports():
    """Test that basic imports work."""
    var x = sqrt(4.0)
    print("sqrt(4.0) =", x)
    
    var width = simdwidthof[DType.float32]()
    print("SIMD width for float32:", width)

fn main():
    """Test available imports."""
    print("🧪 Testing Available Imports")
    print("===========================")
    
    test_basic_imports()
    
    print("✅ Basic imports work!")
    print("❌ Tensor imports need to be investigated")