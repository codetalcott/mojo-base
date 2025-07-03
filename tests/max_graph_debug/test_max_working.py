#!/usr/bin/env python3
"""
Working MAX Graph implementation based on official patterns
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
import numpy as np

def test_working_constant():
    """Test constant creation using the correct pattern."""
    print("=== Testing Working Constant Creation ===")
    
    try:
        # Use the pattern from MAX source code
        const_int = ops.constant(1, DType.int64, device=DeviceRef.CPU())
        print("✅ Integer constant created successfully")
        
        const_float = ops.constant(1.5, DType.float32, device=DeviceRef.CPU())
        print("✅ Float constant created successfully")
        
        # Test array constant
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        const_array = ops.constant(arr, DType.float32, device=DeviceRef.CPU())
        print("✅ Array constant created successfully")
        
        return const_array
        
    except Exception as e:
        print(f"❌ Constant creation failed: {e}")
        return None

def test_graph_decorator():
    """Test graph creation using decorator pattern."""
    print("\n=== Testing Graph Decorator Pattern ===")
    
    try:
        # Simple computation graph
        def simple_computation():
            a = ops.constant(2.0, DType.float32, device=DeviceRef.CPU())
            b = ops.constant(3.0, DType.float32, device=DeviceRef.CPU())
            c = ops.add(a, b)
            return c
        
        # Create graph
        graph = g.Graph(
            name="simple_test",
            forward=simple_computation
        )
        
        print("✅ Graph created with decorator pattern")
        print(f"   Graph type: {type(graph)}")
        print(f"   Graph methods: {[x for x in dir(graph) if not x.startswith('_')]}")
        
        return graph
        
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        return None

def test_parameterized_graph():
    """Test graph with input parameters."""
    print("\n=== Testing Parameterized Graph ===")
    
    try:
        # Define input types
        input_type = TensorType(DType.float32, [2, 3], DeviceRef.CPU())
        
        def param_forward(x):
            # Square the input
            squared = ops.mul(x, x)
            # Sum along last axis
            result = ops.sum(squared, axis=1)
            return result
        
        # Create parameterized graph
        param_graph = g.Graph(
            name="param_test",
            forward=param_forward,
            input_types=[input_type]
        )
        
        print("✅ Parameterized graph created successfully")
        return param_graph
        
    except Exception as e:
        print(f"❌ Parameterized graph failed: {e}")
        return None

def create_semantic_search_graph():
    """Create a working semantic search graph."""
    print("\n=== Creating Semantic Search Graph ===")
    
    try:
        # Define input types for semantic search
        query_type = TensorType(DType.float32, [1, 768], DeviceRef.CPU())      # Single query
        corpus_type = TensorType(DType.float32, [1000, 768], DeviceRef.CPU())  # Corpus vectors
        
        def semantic_search_forward(query, corpus):
            # L2 normalization implementation
            def l2_normalize(tensor, axis=1):
                # Square the tensor
                squared = ops.mul(tensor, tensor)
                # Sum along the specified axis
                sum_squared = ops.sum(squared, axis=axis)
                # Take square root to get L2 norm
                norm = ops.sqrt(sum_squared)
                # Add small epsilon for numerical stability
                epsilon = ops.constant(1e-8, DType.float32, device=DeviceRef.CPU())
                norm_stable = ops.add(norm, epsilon)
                # Normalize by dividing by the norm
                return ops.div(tensor, norm_stable)
            
            # Normalize both query and corpus vectors
            query_normalized = l2_normalize(query, axis=1)
            corpus_normalized = l2_normalize(corpus, axis=1)
            
            # Compute cosine similarity: query @ corpus.T
            # Transpose corpus: [1000, 768] -> [768, 1000]
            corpus_transposed = ops.transpose(corpus_normalized, axis_1=0, axis_2=1)
            
            # Matrix multiplication: [1, 768] @ [768, 1000] -> [1, 1000]
            similarities = ops.matmul(query_normalized, corpus_transposed)
            
            return similarities
        
        # Create the semantic search graph
        search_graph = g.Graph(
            name="semantic_search",
            forward=semantic_search_forward,
            input_types=[query_type, corpus_type]
        )
        
        print("✅ Semantic search graph created successfully!")
        print(f"   Input types: query {query_type}, corpus {corpus_type}")
        
        return search_graph
        
    except Exception as e:
        print(f"❌ Semantic search graph failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("=== MAX Graph Working Implementation Test ===")
    print()
    
    # Test basic constant creation
    const = test_working_constant()
    
    # Test simple graph creation
    simple_graph = test_graph_decorator()
    
    # Test parameterized graph
    param_graph = test_parameterized_graph()
    
    # Test semantic search graph
    search_graph = create_semantic_search_graph()
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Constants: {'✅' if const is not None else '❌'}")
    print(f"Simple graph: {'✅' if simple_graph is not None else '❌'}")
    print(f"Parameterized graph: {'✅' if param_graph is not None else '❌'}")
    print(f"Semantic search graph: {'✅' if search_graph is not None else '❌'}")
    
    if search_graph is not None:
        print(f"\n🎉 SUCCESS: MAX Graph semantic search is ready!")
        return search_graph
    else:
        print(f"\n❌ FAILED: Need to debug MAX Graph issues")
        return None

if __name__ == "__main__":
    main()