#!/usr/bin/env python3
"""
Explore MAX Graph API to understand proper usage patterns
"""

import max.graph as g
import numpy as np

def explore_graph_api():
    """Systematically explore the MAX Graph API."""
    print("=== MAX Graph API Exploration ===")
    print()
    
    # 1. Explore Graph class
    print("1. Graph class methods:")
    graph_methods = [x for x in dir(g.Graph) if not x.startswith('_')]
    for method in sorted(graph_methods):
        print(f"   - {method}")
    print()
    
    # 2. Explore TensorType
    print("2. TensorType class methods:")
    tensor_methods = [x for x in dir(g.TensorType) if not x.startswith('_')]
    for method in sorted(tensor_methods):
        print(f"   - {method}")
    print()
    
    # 3. Explore operations
    print("3. Available operations:")
    ops_list = [x for x in dir(g.ops) if not x.startswith('_') and not x.isupper()]
    for op in sorted(ops_list):
        print(f"   - {op}")
    print()
    
    # 4. Explore type system
    print("4. Type system:")
    print(f"   - DTypes: {[x for x in dir(g.type.DType) if not x.startswith('_')]}")
    print(f"   - DeviceRef: {[x for x in dir(g.type.DeviceRef) if not x.startswith('_')]}")
    print()

def test_basic_graph_creation():
    """Test basic graph creation patterns."""
    print("=== Testing Basic Graph Creation ===")
    print()
    
    try:
        # Test 1: Simple graph with constants
        print("Test 1: Creating graph with constants...")
        
        def simple_forward():
            a = g.ops.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
            b = g.ops.constant(np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32))
            c = g.ops.add(a, b)
            return c
        
        simple_graph = g.Graph(
            name="simple_test",
            forward=simple_forward
        )
        print("   ✅ Simple graph created successfully")
        
    except Exception as e:
        print(f"   ❌ Simple graph failed: {e}")
    
    try:
        # Test 2: Graph with input parameters
        print("Test 2: Creating graph with input parameters...")
        
        dtype = g.type.DType.float32
        device = g.type.DeviceRef.CPU
        
        input_type = g.TensorType(dtype, [2, 2], device)
        
        def param_forward(x):
            y = g.ops.mul(x, x)  # Square the input
            return y
        
        param_graph = g.Graph(
            name="param_test",
            forward=param_forward,
            input_types=[input_type]
        )
        print("   ✅ Parameterized graph created successfully")
        
    except Exception as e:
        print(f"   ❌ Parameterized graph failed: {e}")

def test_semantic_search_pattern():
    """Test the specific pattern we need for semantic search."""
    print("=== Testing Semantic Search Pattern ===")
    print()
    
    try:
        print("Creating semantic search graph...")
        
        # Define types
        dtype = g.type.DType.float32
        device = g.type.DeviceRef.CPU
        
        # Query: [1, 768] - single query vector
        query_type = g.TensorType(dtype, [1, 768], device)
        # Corpus: [1000, 768] - corpus of vectors  
        corpus_type = g.TensorType(dtype, [1000, 768], device)
        
        def semantic_search_forward(query, corpus):
            # L2 normalization
            def l2_normalize(tensor, axis=1):
                squared = g.ops.mul(tensor, tensor)
                sum_squared = g.ops.sum(squared, axis=axis)
                norm = g.ops.sqrt(sum_squared)
                # Add epsilon for numerical stability
                epsilon = g.ops.constant(np.array(1e-8, dtype=np.float32))
                norm_stable = g.ops.add(norm, epsilon)
                return g.ops.div(tensor, norm_stable)
            
            # Normalize vectors
            query_norm = l2_normalize(query)
            corpus_norm = l2_normalize(corpus)
            
            # Compute similarities: query @ corpus.T
            corpus_t = g.ops.transpose(corpus_norm, axis_1=0, axis_2=1)
            similarities = g.ops.matmul(query_norm, corpus_t)
            
            return similarities
        
        search_graph = g.Graph(
            name="semantic_search",
            forward=semantic_search_forward,
            input_types=[query_type, corpus_type]
        )
        
        print("   ✅ Semantic search graph created successfully!")
        return search_graph
        
    except Exception as e:
        print(f"   ❌ Semantic search graph failed: {e}")
        return None

def test_graph_compilation(graph):
    """Test graph compilation."""
    if graph is None:
        print("   ⚠️ No graph to compile")
        return
        
    print("=== Testing Graph Compilation ===")
    print()
    
    try:
        print("Compiling graph...")
        
        # Check if graph has compile method
        if hasattr(graph, 'compile'):
            print("   Found compile method, attempting compilation...")
            graph.compile()
            print("   ✅ Graph compiled successfully!")
        else:
            print("   ⚠️ Graph doesn't have compile method")
            print(f"   Available methods: {[x for x in dir(graph) if not x.startswith('_')]}")
            
    except Exception as e:
        print(f"   ❌ Graph compilation failed: {e}")

def main():
    """Main exploration function."""
    explore_graph_api()
    test_basic_graph_creation()
    search_graph = test_semantic_search_pattern()
    test_graph_compilation(search_graph)
    
    print("\n=== Exploration Complete ===")
    print("Use this information to implement correct MAX Graph integration")

if __name__ == "__main__":
    main()