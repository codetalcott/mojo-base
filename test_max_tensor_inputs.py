#!/usr/bin/env python3
"""
Test correct tensor input format for MAX Graph execution
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession
import numpy as np
import time

def test_tensor_input_formats():
    """Test different ways to pass tensor inputs to model.execute()"""
    print("=== Testing Tensor Input Formats ===")
    
    # Create simple graph
    dtype = DType.float32
    device = DeviceRef.CPU()
    
    query_type = TensorType(dtype, [1, 768], device)
    corpus_type = TensorType(dtype, [100, 768], device)
    
    def forward(query, corpus):
        corpus_t = ops.transpose(corpus, axis_1=0, axis_2=1)
        return ops.matmul(query, corpus_t)
    
    graph = g.Graph(
        name="tensor_format_test",
        forward=forward,
        input_types=[query_type, corpus_type]
    )
    
    # Load graph
    session = InferenceSession()
    model = session.load(graph)
    print("✅ Model loaded successfully")
    
    # Prepare test data
    query_np = np.random.randn(1, 768).astype(np.float32)
    corpus_np = np.random.randn(100, 768).astype(np.float32)
    
    # Test 1: Individual arguments (not list)
    print("\nTest 1: Individual arguments...")
    try:
        result = model.execute(query_np, corpus_np)
        print("✅ Individual arguments work!")
        return model, query_np, corpus_np
    except Exception as e:
        print(f"❌ Individual arguments failed: {e}")
    
    # Test 2: Check if we need max.driver.Tensor
    print("\nTest 2: Investigating max.driver.Tensor...")
    try:
        import max.driver as driver
        print("   max.driver available")
        
        # Check if there's a Tensor class
        if hasattr(driver, 'Tensor'):
            print("   max.driver.Tensor exists")
            
            # Try converting numpy arrays
            query_tensor = driver.Tensor(query_np)
            corpus_tensor = driver.Tensor(corpus_np)
            print("   ✅ Tensors created")
            
            result = model.execute(query_tensor, corpus_tensor)
            print("✅ max.driver.Tensor format works!")
            return model, query_tensor, corpus_tensor
            
        else:
            print("   max.driver.Tensor not found")
            
    except ImportError:
        print("   max.driver not available")
    except Exception as e:
        print(f"   max.driver.Tensor failed: {e}")
    
    # Test 3: Check for MojoValue
    print("\nTest 3: Investigating MojoValue...")
    try:
        from max.engine import MojoValue
        print("   MojoValue available")
        
        # Try converting
        query_mojo = MojoValue(query_np)
        corpus_mojo = MojoValue(corpus_np)
        print("   ✅ MojoValues created")
        
        result = model.execute(query_mojo, corpus_mojo)
        print("✅ MojoValue format works!")
        return model, query_mojo, corpus_mojo
        
    except Exception as e:
        print(f"   MojoValue failed: {e}")
    
    # Test 4: DLPack protocol
    print("\nTest 4: DLPack protocol...")
    try:
        # Check if numpy arrays support DLPack
        if hasattr(query_np, '__dlpack__'):
            query_dlpack = query_np.__dlpack__()
            corpus_dlpack = corpus_np.__dlpack__()
            print("   ✅ DLPack conversion successful")
            
            result = model.execute(query_dlpack, corpus_dlpack)
            print("✅ DLPack format works!")
            return model, query_dlpack, corpus_dlpack
        else:
            print("   Numpy arrays don't have __dlpack__")
            
    except Exception as e:
        print(f"   DLPack failed: {e}")
    
    return None, None, None

def investigate_model_signature():
    """Investigate model signature and input requirements."""
    print("\n=== Model Signature Investigation ===")
    
    # Create simple graph
    dtype = DType.float32
    device = DeviceRef.CPU()
    
    input_type = TensorType(dtype, [2, 3], device)
    
    def simple_forward(x):
        return ops.add(x, x)
    
    graph = g.Graph(
        name="signature_test",
        forward=simple_forward,
        input_types=[input_type]
    )
    
    session = InferenceSession()
    model = session.load(graph)
    
    # Check model signature
    print("Model signature:")
    if hasattr(model, 'signature'):
        sig = model.signature()
        print(f"   {sig}")
    
    # Check input metadata
    if hasattr(model, 'input_metadata'):
        input_meta = model.input_metadata()
        print(f"Input metadata: {input_meta}")
    
    # Check output metadata
    if hasattr(model, 'output_metadata'):
        output_meta = model.output_metadata()
        print(f"Output metadata: {output_meta}")
    
    return model

def test_working_execution():
    """Test execution with the working tensor format."""
    print("\n=== Testing Working Execution ===")
    
    # Find working tensor format
    model, query_tensor, corpus_tensor = test_tensor_input_formats()
    
    if model is not None:
        print(f"✅ Found working tensor format!")
        
        # Benchmark performance
        print("\nBenchmarking performance...")
        times = []
        
        for i in range(5):
            start_time = time.time()
            result = model.execute(query_tensor, corpus_tensor)
            exec_time = (time.time() - start_time) * 1000
            times.append(exec_time)
            print(f"   Run {i+1}: {exec_time:.3f}ms")
        
        avg_time = sum(times) / len(times)
        print(f"\n📊 Performance Summary:")
        print(f"   Average: {avg_time:.3f}ms")
        print(f"   Best: {min(times):.3f}ms")
        print(f"   Worst: {max(times):.3f}ms")
        
        # Check result format
        print(f"\nResult analysis:")
        print(f"   Type: {type(result)}")
        if hasattr(result, '__len__'):
            print(f"   Length: {len(result)}")
            if len(result) > 0:
                first = result[0] if hasattr(result, '__getitem__') else result
                print(f"   First item type: {type(first)}")
                if hasattr(first, 'shape'):
                    print(f"   Shape: {first.shape}")
        
        return True, avg_time
    
    else:
        print("❌ No working tensor format found")
        return False, None

def main():
    """Main test function."""
    print("🔧 Testing MAX Graph Tensor Input Formats")
    print("=" * 60)
    
    # Investigate model requirements
    model = investigate_model_signature()
    
    # Test execution with correct format
    success, avg_time = test_working_execution()
    
    if success:
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"✅ MAX Graph execution fully working")
        print(f"✅ Average performance: {avg_time:.3f}ms")
        print(f"✅ Ready to integrate into semantic search")
    else:
        print(f"\n❌ Still need to resolve tensor input format")

if __name__ == "__main__":
    main()