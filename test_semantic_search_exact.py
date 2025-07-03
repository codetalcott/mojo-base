#!/usr/bin/env python3
"""
Test exact semantic search pattern to match our implementation
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
import numpy as np

def test_exact_semantic_search():
    """Test the exact pattern from our semantic search implementation."""
    print("=== Testing Exact Semantic Search Pattern ===")
    
    try:
        # Define input types exactly as in our implementation
        dtype = DType.float32
        device = DeviceRef.CPU()
        
        query_type = TensorType(dtype, [1, 768], device)
        corpus_type = TensorType(dtype, [1000, 768], device)
        
        def forward(query, corpus):
            # L2 normalization implementation - exact copy from our code
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
            # Transpose corpus for matrix multiplication
            corpus_transposed = ops.transpose(corpus_normalized, axis_1=0, axis_2=1)
            
            # Matrix multiplication for similarity computation
            similarities = ops.matmul(query_normalized, corpus_transposed)
            
            return similarities
        
        # Create the graph exactly as in our implementation
        search_graph = g.Graph(
            name="semantic_search_graph",
            forward=forward,
            input_types=[query_type, corpus_type]
        )
        
        print("✅ Exact semantic search graph created successfully!")
        return search_graph
        
    except Exception as e:
        print(f"❌ Exact semantic search failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simplified_version():
    """Test a simplified version without constants."""
    print("\n=== Testing Simplified Version ===")
    
    try:
        dtype = DType.float32
        device = DeviceRef.CPU()
        
        query_type = TensorType(dtype, [1, 768], device)
        corpus_type = TensorType(dtype, [1000, 768], device)
        
        def simple_forward(query, corpus):
            # Simplified L2 norm without epsilon constant
            def simple_l2_normalize(tensor, axis=1):
                squared = ops.mul(tensor, tensor)
                sum_squared = ops.sum(squared, axis=axis)
                norm = ops.sqrt(sum_squared)
                return ops.div(tensor, norm)
            
            query_norm = simple_l2_normalize(query, axis=1)
            corpus_norm = simple_l2_normalize(corpus, axis=1)
            
            corpus_t = ops.transpose(corpus_norm, axis_1=0, axis_2=1)
            similarities = ops.matmul(query_norm, corpus_t)
            
            return similarities
        
        simple_graph = g.Graph(
            name="simple_semantic_search",
            forward=simple_forward,
            input_types=[query_type, corpus_type]
        )
        
        print("✅ Simplified semantic search works!")
        return simple_graph
        
    except Exception as e:
        print(f"❌ Simplified version failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_different_constant_approach():
    """Test different approaches to constant creation."""
    print("\n=== Testing Different Constant Approaches ===")
    
    try:
        dtype = DType.float32
        device = DeviceRef.CPU()
        
        query_type = TensorType(dtype, [1, 768], device)
        corpus_type = TensorType(dtype, [1000, 768], device)
        
        def constant_test_forward(query, corpus):
            def l2_normalize_with_different_epsilon(tensor, axis=1):
                squared = ops.mul(tensor, tensor)
                sum_squared = ops.sum(squared, axis=axis)
                norm = ops.sqrt(sum_squared)
                
                # Try different constant creation approaches
                # Approach 1: Direct scalar
                try:
                    epsilon = ops.constant(1e-8, DType.float32, device=device)
                    norm_stable = ops.add(norm, epsilon)
                    return ops.div(tensor, norm_stable)
                except:
                    # Approach 2: Without device parameter
                    try:
                        epsilon = ops.constant(1e-8, DType.float32)
                        norm_stable = ops.add(norm, epsilon)
                        return ops.div(tensor, norm_stable)
                    except:
                        # Approach 3: Just return without epsilon for now
                        return ops.div(tensor, norm)
            
            query_norm = l2_normalize_with_different_epsilon(query, axis=1)
            corpus_norm = l2_normalize_with_different_epsilon(corpus, axis=1)
            
            corpus_t = ops.transpose(corpus_norm, axis_1=0, axis_2=1)
            similarities = ops.matmul(query_norm, corpus_t)
            
            return similarities
        
        constant_graph = g.Graph(
            name="constant_test_semantic_search",
            forward=constant_test_forward,
            input_types=[query_type, corpus_type]
        )
        
        print("✅ Constant approach test works!")
        return constant_graph
        
    except Exception as e:
        print(f"❌ Constant approach failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("=== Testing Semantic Search Exact Pattern ===")
    
    # Test 1: Exact pattern
    exact_graph = test_exact_semantic_search()
    
    # Test 2: Simplified version
    simple_graph = test_simplified_version()
    
    # Test 3: Different constant approaches
    constant_graph = test_different_constant_approach()
    
    print(f"\n=== Results ===")
    print(f"Exact pattern: {'✅' if exact_graph else '❌'}")
    print(f"Simplified version: {'✅' if simple_graph else '❌'}")
    print(f"Constant approaches: {'✅' if constant_graph else '❌'}")
    
    # Return the working graph
    working_graph = exact_graph or simple_graph or constant_graph
    if working_graph:
        print(f"\n🎉 Working MAX Graph semantic search available!")
        return working_graph
    else:
        print(f"\n❌ No working semantic search pattern found")
        return None

if __name__ == "__main__":
    main()