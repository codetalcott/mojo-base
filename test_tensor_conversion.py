#!/usr/bin/env python3
"""
Test tensor conversion methods
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession
import numpy as np

def test_tensor_conversion():
    """Test different methods to convert max.driver.Tensor to numpy."""
    
    # Create simple graph
    dtype = DType.float32
    device = DeviceRef.CPU()
    
    input_type = TensorType(dtype, [2, 3], device)
    
    def forward(x):
        return ops.add(x, x)
    
    graph = g.Graph(
        name="conversion_test",
        forward=forward,
        input_types=[input_type]
    )
    
    session = InferenceSession()
    model = session.load(graph)
    
    # Test data
    test_input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    
    # Execute
    outputs = model.execute(test_input)
    tensor = outputs[0]
    
    print(f"Original tensor: {tensor}")
    print(f"Tensor type: {type(tensor)}")
    print(f"Tensor shape: {tensor.shape}")
    
    # Test conversion methods
    print("\n=== Testing Conversion Methods ===")
    
    # Method 1: .numpy()
    try:
        if hasattr(tensor, 'numpy'):
            np_result1 = tensor.numpy()
            print(f"1. .numpy(): shape={np_result1.shape}, dtype={np_result1.dtype}")
            print(f"   Content: {np_result1}")
        else:
            print("1. .numpy(): NOT AVAILABLE")
    except Exception as e:
        print(f"1. .numpy(): FAILED - {e}")
    
    # Method 2: np.array()
    try:
        np_result2 = np.array(tensor)
        print(f"2. np.array(): shape={np_result2.shape}, dtype={np_result2.dtype}")
        print(f"   Content: {np_result2}")
    except Exception as e:
        print(f"2. np.array(): FAILED - {e}")
    
    # Method 3: np.asarray()
    try:
        np_result3 = np.asarray(tensor)
        print(f"3. np.asarray(): shape={np_result3.shape}, dtype={np_result3.dtype}")
        print(f"   Content: {np_result3}")
    except Exception as e:
        print(f"3. np.asarray(): FAILED - {e}")
    
    # Method 4: Check for __array__ interface
    try:
        if hasattr(tensor, '__array__'):
            np_result4 = tensor.__array__()
            print(f"4. __array__(): shape={np_result4.shape}, dtype={np_result4.dtype}")
            print(f"   Content: {np_result4}")
        else:
            print("4. __array__(): NOT AVAILABLE")
    except Exception as e:
        print(f"4. __array__(): FAILED - {e}")
    
    # Method 5: Check tensor attributes
    print(f"\nTensor attributes: {[x for x in dir(tensor) if not x.startswith('_')]}")
    
    return tensor

def main():
    """Main test."""
    print("🔧 Testing MAX Tensor to NumPy Conversion")
    print("=" * 50)
    
    tensor = test_tensor_conversion()

if __name__ == "__main__":
    main()