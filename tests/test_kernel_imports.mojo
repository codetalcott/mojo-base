"""
Test kernel imports to verify the correct import paths
"""

# Test the official imports from Modular examples
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from algorithm import parallelize, vectorize
from memory import UnsafePointer
from math import sqrt, exp
from sys import simdwidthof
from random import random_float64

fn test_imports():
    """Test that all imports work correctly."""
    var x = sqrt(4.0)
    var width = simdwidthof[DType.float32]()
    var random_val = random_float64(-1.0, 1.0)
    
    print("✅ sqrt(4.0) =", x)
    print("✅ SIMD width:", width)
    print("✅ Random value:", random_val)

fn main():
    """Test kernel imports."""
    print("🧪 Testing Kernel Imports")
    print("=========================")
    
    test_imports()
    
    print("✅ All basic imports successful!")
    print("🎉 Kernel import verification complete!")