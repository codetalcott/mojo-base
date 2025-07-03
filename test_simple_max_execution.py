#!/usr/bin/env python3
"""
Simple test to find working MAX Graph execution pattern
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession
import numpy as np

def test_simple_execution():
    """Test simple execution with different input formats."""
    print("=== Simple MAX Execution Test ===")
    
    # Create minimal graph
    dtype = DType.float32
    device = DeviceRef.CPU()
    input_type = TensorType(dtype, [2, 3], device)
    
    def simple_forward(x):
        return ops.add(x, x)
    
    graph = g.Graph(
        name="simple_test",
        forward=simple_forward,
        input_types=[input_type]
    )
    
    # Load model
    session = InferenceSession()
    model = session.load(graph)
    print("✅ Model loaded")
    
    # Test data
    test_input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"Test input shape: {test_input.shape}")
    
    # Test different input methods
    print("\n1. Testing direct numpy array...")
    try:
        result = model.execute(test_input)
        print("✅ Direct numpy works!")
        print(f"   Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Direct numpy failed: {e}")
    
    print("\n2. Testing with max.driver...")
    try:
        import max.driver as driver
        
        # Check available classes
        driver_classes = [x for x in dir(driver) if not x.startswith('_')]
        print(f"   Available: {driver_classes}")
        
        if 'Tensor' in driver_classes:
            tensor = driver.Tensor(test_input)
            result = model.execute(tensor)
            print("✅ max.driver.Tensor works!")
            print(f"   Result: {result}")
            return True
            
    except Exception as e:
        print(f"❌ max.driver failed: {e}")
    
    print("\n3. Testing different numpy dtypes...")
    try:
        # Ensure exact dtype match
        test_input_f32 = test_input.astype(np.float32)
        result = model.execute(test_input_f32)
        print("✅ Explicit float32 works!")
        print(f"   Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Explicit float32 failed: {e}")
    
    print("\n4. Testing contiguous arrays...")
    try:
        test_input_contig = np.ascontiguousarray(test_input)
        result = model.execute(test_input_contig)
        print("✅ Contiguous array works!")
        print(f"   Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Contiguous array failed: {e}")
    
    return False

def main():
    """Main test."""
    print("🧪 Simple MAX Execution Test")
    print("=" * 40)
    
    success = test_simple_execution()
    
    if success:
        print(f"\n🎉 SUCCESS: Found working execution pattern!")
    else:
        print(f"\n❌ No working pattern found")

if __name__ == "__main__":
    main()