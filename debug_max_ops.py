#!/usr/bin/env python3
"""
Debug MAX Graph operations to isolate the issue
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
import numpy as np

def test_individual_operations():
    """Test each operation individually to find the issue."""
    print("=== Testing Individual Operations ===")
    
    try:
        # Define input types
        dtype = DType.float32
        device = DeviceRef.CPU()
        tensor_type = TensorType(dtype, [2, 3], device)
        
        def test_mul(x):
            return ops.mul(x, x)
        
        mul_graph = g.Graph(
            name="test_mul",
            forward=test_mul,
            input_types=[tensor_type]
        )
        print("✅ Multiplication operation works")
        
    except Exception as e:
        print(f"❌ Multiplication failed: {e}")
    
    try:
        def test_sum(x):
            return ops.sum(x, axis=1)
        
        sum_graph = g.Graph(
            name="test_sum", 
            forward=test_sum,
            input_types=[tensor_type]
        )
        print("✅ Sum operation works")
        
    except Exception as e:
        print(f"❌ Sum failed: {e}")
    
    try:
        def test_sum_default(x):
            return ops.sum(x)  # Use default axis
        
        sum_default_graph = g.Graph(
            name="test_sum_default",
            forward=test_sum_default,
            input_types=[tensor_type]
        )
        print("✅ Sum with default axis works")
        
    except Exception as e:
        print(f"❌ Sum with default axis failed: {e}")
    
    try:
        def test_sqrt(x):
            return ops.sqrt(x)
        
        sqrt_graph = g.Graph(
            name="test_sqrt",
            forward=test_sqrt,
            input_types=[tensor_type]
        )
        print("✅ Sqrt operation works")
        
    except Exception as e:
        print(f"❌ Sqrt failed: {e}")

def test_l2_norm_step_by_step():
    """Test L2 normalization step by step."""
    print("\n=== Testing L2 Normalization Steps ===")
    
    try:
        dtype = DType.float32
        device = DeviceRef.CPU()
        tensor_type = TensorType(dtype, [2, 3], device)
        
        # Step 1: Square
        def test_square(x):
            return ops.mul(x, x)
        
        square_graph = g.Graph(
            name="test_square",
            forward=test_square,
            input_types=[tensor_type]
        )
        print("✅ Step 1: Square works")
        
        # Step 2: Sum along axis
        def test_sum_axis(x):
            squared = ops.mul(x, x)
            return ops.sum(squared, axis=1)
        
        sum_axis_graph = g.Graph(
            name="test_sum_axis",
            forward=test_sum_axis,
            input_types=[tensor_type]
        )
        print("✅ Step 2: Sum along axis works")
        
        # Step 3: Sqrt
        def test_sqrt_after_sum(x):
            squared = ops.mul(x, x)
            summed = ops.sum(squared, axis=1)
            return ops.sqrt(summed)
        
        sqrt_after_sum_graph = g.Graph(
            name="test_sqrt_after_sum",
            forward=test_sqrt_after_sum,
            input_types=[tensor_type]
        )
        print("✅ Step 3: Sqrt after sum works")
        
        # Step 4: Add epsilon
        def test_add_epsilon(x):
            squared = ops.mul(x, x)
            summed = ops.sum(squared, axis=1)
            norm = ops.sqrt(summed)
            # Use scalar constant
            epsilon = ops.constant(1e-8, DType.float32, device=DeviceRef.CPU())
            return ops.add(norm, epsilon)
        
        add_epsilon_graph = g.Graph(
            name="test_add_epsilon",
            forward=test_add_epsilon,
            input_types=[tensor_type]
        )
        print("✅ Step 4: Add epsilon works")
        
        # Step 5: Division
        def test_full_l2_norm(x):
            squared = ops.mul(x, x)
            summed = ops.sum(squared, axis=1)
            norm = ops.sqrt(summed)
            epsilon = ops.constant(1e-8, DType.float32, device=DeviceRef.CPU())
            norm_stable = ops.add(norm, epsilon)
            return ops.div(x, norm_stable)
        
        full_l2_graph = g.Graph(
            name="test_full_l2_norm",
            forward=test_full_l2_norm,
            input_types=[tensor_type]
        )
        print("✅ Step 5: Full L2 normalization works")
        
        return full_l2_graph
        
    except Exception as e:
        print(f"❌ L2 normalization failed at some step: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main debug function."""
    print("=== MAX Graph Operations Debug ===")
    test_individual_operations()
    l2_graph = test_l2_norm_step_by_step()
    
    if l2_graph:
        print(f"\n🎉 L2 normalization graph created successfully!")
    else:
        print(f"\n❌ L2 normalization has issues")

if __name__ == "__main__":
    main()