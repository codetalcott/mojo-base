#!/usr/bin/env python3
"""
Debug MAX Graph result shape issue
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession
import numpy as np

def debug_max_result_shape():
    """Debug the shape issue in MAX Graph results."""
    print("=== Debugging MAX Graph Result Shape ===")
    
    # Create semantic search graph (same as our implementation)
    dtype = DType.float32
    device = DeviceRef.CPU()
    
    query_type = TensorType(dtype, [1, 768], device)
    corpus_type = TensorType(dtype, [100, 768], device)  # Smaller for debugging
    
    def forward(query, corpus):
        # Same as our semantic search
        corpus_t = ops.transpose(corpus, axis_1=0, axis_2=1)
        similarities = ops.matmul(query, corpus_t)
        return similarities
    
    graph = g.Graph(
        name="debug_semantic_search",
        forward=forward,
        input_types=[query_type, corpus_type]
    )
    
    # Load and execute
    session = InferenceSession()
    model = session.load(graph)
    
    # Test data
    query = np.random.randn(1, 768).astype(np.float32)
    corpus = np.random.randn(100, 768).astype(np.float32)
    
    print(f"Input shapes: query {query.shape}, corpus {corpus.shape}")
    
    # Execute
    outputs = model.execute(query, corpus)
    
    print(f"Raw output type: {type(outputs)}")
    print(f"Raw output length: {len(outputs) if hasattr(outputs, '__len__') else 'no len'}")
    
    if hasattr(outputs, '__len__') and len(outputs) > 0:
        first_output = outputs[0]
        print(f"First output type: {type(first_output)}")
        print(f"First output: {first_output}")
        
        # Try different ways to get the shape
        if hasattr(first_output, 'shape'):
            print(f"First output shape: {first_output.shape}")
        
        # Try conversion to numpy
        if hasattr(first_output, 'numpy'):
            np_result = first_output.numpy()
            print(f"Numpy conversion shape: {np_result.shape}")
            print(f"Numpy result sample: {np_result}")
        elif hasattr(first_output, '__array__'):
            np_result = np.array(first_output)
            print(f"Array conversion shape: {np_result.shape}")
            print(f"Array result sample: {np_result}")
    
    # Test expected shapes
    expected_shape = (1, 100)  # [1, 768] @ [768, 100] = [1, 100]
    print(f"Expected result shape: {expected_shape}")

def test_matrix_mult_shapes():
    """Test matrix multiplication shape behavior specifically."""
    print("\n=== Testing Matrix Multiplication Shapes ===")
    
    # Test simple matrix multiplication
    dtype = DType.float32
    device = DeviceRef.CPU()
    
    a_type = TensorType(dtype, [2, 3], device)
    b_type = TensorType(dtype, [3, 4], device)
    
    def matmul_forward(a, b):
        return ops.matmul(a, b)
    
    graph = g.Graph(
        name="test_matmul",
        forward=matmul_forward,
        input_types=[a_type, b_type]
    )
    
    session = InferenceSession()
    model = session.load(graph)
    
    # Test data
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 4).astype(np.float32)
    
    print(f"Input shapes: a {a.shape}, b {b.shape}")
    print(f"Expected result shape: {(2, 4)}")
    
    outputs = model.execute(a, b)
    result = outputs[0]
    
    print(f"Result type: {type(result)}")
    if hasattr(result, 'shape'):
        print(f"Result shape: {result.shape}")
    
    if hasattr(result, 'numpy'):
        np_result = result.numpy()
        print(f"Numpy result shape: {np_result.shape}")

def main():
    """Main debug function."""
    print("🔍 Debugging MAX Graph Result Shapes")
    print("=" * 50)
    
    debug_max_result_shape()
    test_matrix_mult_shapes()

if __name__ == "__main__":
    main()