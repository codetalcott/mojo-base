#!/usr/bin/env python3
"""
Test MAX Graph constant creation and basic operations
"""

import max.graph as g
import numpy as np

def test_constant_creation():
    """Test different ways to create constants."""
    print("=== Testing Constant Creation ===")
    
    try:
        dtype = g.type.DType.float32
        device = g.type.DeviceRef.CPU
        
        # Test simple constant
        print("Creating simple constant...")
        const = g.ops.constant(np.array([1.0, 2.0], dtype=np.float32), dtype, device)
        print("✅ Simple constant created successfully")
        print(f"   Type: {type(const)}")
        
        # Test 2D constant  
        print("Creating 2D constant...")
        const2d = g.ops.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), dtype, device)
        print("✅ 2D constant created successfully")
        
        return const, const2d
        
    except Exception as e:
        print(f"❌ Constant creation failed: {e}")
        return None, None

def test_operations(const, const2d):
    """Test basic operations."""
    if const is None or const2d is None:
        print("⚠️ No constants to test operations")
        return
        
    print("\n=== Testing Operations ===")
    
    try:
        # Test add
        print("Testing add operation...")
        result = g.ops.add(const, const)
        print("✅ Add operation successful")
        
        # Test matrix multiply
        print("Testing matrix multiply...")
        result2 = g.ops.matmul(const2d, const2d)
        print("✅ Matrix multiply successful")
        
        return result, result2
        
    except Exception as e:
        print(f"❌ Operations failed: {e}")
        return None, None

def test_graph_with_constants():
    """Test creating a graph with constants (no parameters)."""
    print("\n=== Testing Graph with Constants ===")
    
    try:
        dtype = g.type.DType.float32
        device = g.type.DeviceRef.CPU
        
        @g.graph.current
        def constant_graph():
            a = g.ops.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), dtype, device)
            b = g.ops.constant(np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32), dtype, device)
            c = g.ops.add(a, b)
            return c
        
        result = constant_graph()
        print("✅ Graph with constants created successfully")
        print(f"   Result type: {type(result)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Graph with constants failed: {e}")
        return None

def main():
    """Main test function."""
    const, const2d = test_constant_creation()
    test_operations(const, const2d) 
    test_graph_with_constants()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()