#!/usr/bin/env python3
"""
Test the CORRECT MAX Graph execution pattern:
session.load() returns Model, and Model.execute() runs inference
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession
import numpy as np
import time

def test_correct_execution_pattern():
    """Test the correct execution pattern: session.load() -> model.execute()"""
    print("=== Testing CORRECT MAX Execution Pattern ===")
    
    # Create semantic search graph
    dtype = DType.float32
    device = DeviceRef.CPU()
    
    query_type = TensorType(dtype, [1, 768], device)
    corpus_type = TensorType(dtype, [100, 768], device)  # Smaller for testing
    
    def semantic_forward(query, corpus):
        # Transpose corpus for matrix multiplication
        corpus_t = ops.transpose(corpus, axis_1=0, axis_2=1)
        # Compute similarities
        similarities = ops.matmul(query, corpus_t)
        return similarities
    
    graph = g.Graph(
        name="working_semantic_search",
        forward=semantic_forward,
        input_types=[query_type, corpus_type]
    )
    
    print("✅ Semantic search graph created")
    
    try:
        # Step 1: Create InferenceSession
        session = InferenceSession()
        print("✅ InferenceSession created")
        
        # Step 2: Load graph and get Model back
        model = session.load(graph)
        print("✅ Graph loaded, Model returned")
        print(f"   Model type: {type(model)}")
        print(f"   Model methods: {[x for x in dir(model) if not x.startswith('_')]}")
        
        # Step 3: Prepare test data
        query = np.random.randn(1, 768).astype(np.float32)
        corpus = np.random.randn(100, 768).astype(np.float32)
        
        print(f"✅ Test data prepared: query {query.shape}, corpus {corpus.shape}")
        
        # Step 4: Execute using Model.execute()
        start_time = time.time()
        result = model.execute([query, corpus])
        execution_time = (time.time() - start_time) * 1000
        
        print(f"✅ Execution successful!")
        print(f"   Execution time: {execution_time:.3f}ms")
        print(f"   Result type: {type(result)}")
        print(f"   Result length: {len(result) if hasattr(result, '__len__') else 'unknown'}")
        
        if hasattr(result, '__len__') and len(result) > 0:
            first_result = result[0]
            print(f"   First result type: {type(first_result)}")
            if hasattr(first_result, 'shape'):
                print(f"   First result shape: {first_result.shape}")
        
        return model, execution_time
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_performance_comparison():
    """Compare MAX Graph performance with different configurations."""
    print("\n=== Performance Comparison Test ===")
    
    configs = [
        {"corpus_size": 100, "name": "small"},
        {"corpus_size": 1000, "name": "medium"},
        {"corpus_size": 5000, "name": "large"}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']} corpus ({config['corpus_size']} vectors)...")
        
        # Create graph for this configuration
        dtype = DType.float32
        device = DeviceRef.CPU()
        
        query_type = TensorType(dtype, [1, 768], device)
        corpus_type = TensorType(dtype, [config['corpus_size'], 768], device)
        
        def forward(query, corpus):
            corpus_t = ops.transpose(corpus, axis_1=0, axis_2=1)
            return ops.matmul(query, corpus_t)
        
        graph = g.Graph(
            name=f"perf_test_{config['name']}",
            forward=forward,
            input_types=[query_type, corpus_type]
        )
        
        try:
            # Load and execute
            session = InferenceSession()
            model = session.load(graph)
            
            # Prepare data
            query = np.random.randn(1, 768).astype(np.float32)
            corpus = np.random.randn(config['corpus_size'], 768).astype(np.float32)
            
            # Benchmark execution
            times = []
            for i in range(3):  # 3 runs for average
                start_time = time.time()
                result = model.execute([query, corpus])
                exec_time = (time.time() - start_time) * 1000
                times.append(exec_time)
            
            avg_time = sum(times) / len(times)
            results.append({
                'config': config['name'],
                'corpus_size': config['corpus_size'],
                'avg_time_ms': avg_time,
                'times': times
            })
            
            print(f"   ✅ Average: {avg_time:.3f}ms (runs: {[f'{t:.1f}' for t in times]})")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'config': config['name'],
                'corpus_size': config['corpus_size'],
                'avg_time_ms': None,
                'error': str(e)
            })
    
    return results

def test_advanced_operations():
    """Test MAX Graph with more complex operations (L2 normalization, etc.)"""
    print("\n=== Advanced Operations Test ===")
    
    dtype = DType.float32
    device = DeviceRef.CPU()
    
    query_type = TensorType(dtype, [1, 768], device)
    corpus_type = TensorType(dtype, [1000, 768], device)
    
    def advanced_forward(query, corpus):
        # L2 normalization (simplified - without epsilon for testing)
        def l2_normalize(tensor):
            squared = ops.mul(tensor, tensor)
            sum_squared = ops.sum(squared, 1)  # Sum along dim 1
            norm = ops.sqrt(sum_squared)
            return ops.div(tensor, norm)
        
        # Normalize both tensors
        query_norm = l2_normalize(query)
        corpus_norm = l2_normalize(corpus)
        
        # Compute cosine similarity
        corpus_t = ops.transpose(corpus_norm, axis_1=0, axis_2=1)
        similarities = ops.matmul(query_norm, corpus_t)
        
        return similarities
    
    graph = g.Graph(
        name="advanced_semantic_search",
        forward=advanced_forward,
        input_types=[query_type, corpus_type]
    )
    
    try:
        print("Testing advanced operations (L2 norm + cosine similarity)...")
        
        session = InferenceSession()
        model = session.load(graph)
        
        # Test data
        query = np.random.randn(1, 768).astype(np.float32)
        corpus = np.random.randn(1000, 768).astype(np.float32)
        
        start_time = time.time()
        result = model.execute([query, corpus])
        exec_time = (time.time() - start_time) * 1000
        
        print(f"✅ Advanced operations successful!")
        print(f"   Execution time: {exec_time:.3f}ms")
        
        return True, exec_time
        
    except Exception as e:
        print(f"❌ Advanced operations failed: {e}")
        # Print the specific error for L2 normalization
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Main test function."""
    print("🚀 Testing WORKING MAX Graph Execution Pattern")
    print("=" * 60)
    print("Key insight: session.load() returns Model, Model.execute() runs inference")
    print()
    
    # Test 1: Basic execution pattern
    model, exec_time = test_correct_execution_pattern()
    
    if model is not None:
        print(f"\n🎉 SUCCESS: MAX Graph execution is WORKING!")
        print(f"   Basic execution time: {exec_time:.3f}ms")
        
        # Test 2: Performance comparison
        perf_results = test_performance_comparison()
        
        # Test 3: Advanced operations
        advanced_works, advanced_time = test_advanced_operations()
        
        # Summary
        print(f"\n📊 Final Results")
        print("=" * 30)
        print(f"✅ Basic execution: {exec_time:.3f}ms")
        
        for result in perf_results:
            if result.get('avg_time_ms'):
                print(f"✅ {result['config']} corpus ({result['corpus_size']}): {result['avg_time_ms']:.3f}ms")
            else:
                print(f"❌ {result['config']} corpus: {result.get('error', 'failed')}")
        
        if advanced_works:
            print(f"✅ Advanced operations: {advanced_time:.3f}ms")
        else:
            print(f"❌ Advanced operations: failed")
        
        print(f"\n🎯 BREAKTHROUGH: MAX Graph execution is now working!")
        print(f"   Ready to integrate into semantic search implementation")
        
    else:
        print(f"\n❌ Basic execution failed - need to investigate further")

if __name__ == "__main__":
    main()