#!/usr/bin/env python3
"""
Test the correct MAX Graph execution pattern based on investigation findings
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession
import numpy as np

def test_inference_session_load_pattern():
    """Test InferenceSession.load() pattern."""
    print("=== Testing InferenceSession.load() Pattern ===")
    
    # Create simple graph
    dtype = DType.float32
    device = DeviceRef.CPU()
    input_type = TensorType(dtype, [2, 3], device)
    
    def simple_forward(x):
        return ops.add(x, x)
    
    test_graph = g.Graph(
        name="load_test",
        forward=simple_forward,
        input_types=[input_type]
    )
    
    print("✅ Graph created")
    
    # Test approach 1: Create empty session, then load
    try:
        print("Approach 1: Empty session + load()")
        session = InferenceSession()
        print("   ✅ Empty InferenceSession created")
        
        # Try loading the graph
        session.load(test_graph)
        print("   ✅ Graph loaded successfully!")
        return session
        
    except Exception as e:
        print(f"   ❌ Load approach failed: {e}")
    
    # Test approach 2: Session with explicit parameters
    try:
        print("Approach 2: Session with parameters + load()")
        session = InferenceSession(num_threads=1)
        print("   ✅ Parameterized InferenceSession created")
        
        session.load(test_graph)
        print("   ✅ Graph loaded successfully!")
        return session
        
    except Exception as e:
        print(f"   ❌ Parameterized load failed: {e}")
    
    return None

def test_graph_serialization():
    """Test if graph needs to be serialized first."""
    print("\n=== Testing Graph Serialization ===")
    
    dtype = DType.float32
    device = DeviceRef.CPU()
    input_type = TensorType(dtype, [2, 3], device)
    
    def simple_forward(x):
        return ops.add(x, x)
    
    # Try with MLIR context
    try:
        print("Approach 1: Graph with MLIR context")
        
        # Check if we can import mlir
        try:
            import mlir
            context = mlir.Context()
            print("   ✅ MLIR context created")
            
            test_graph = g.Graph(
                name="mlir_test",
                forward=simple_forward,
                input_types=[input_type],
                context=context
            )
            print("   ✅ Graph with MLIR context created")
            
            # Try loading
            session = InferenceSession()
            session.load(test_graph)
            print("   ✅ Graph with context loaded!")
            return session
            
        except ImportError:
            print("   ⚠️  MLIR module not directly available")
            
    except Exception as e:
        print(f"   ❌ MLIR context approach failed: {e}")
    
    # Try output/compilation approach
    try:
        print("Approach 2: Graph output investigation")
        
        test_graph = g.Graph(
            name="output_test",
            forward=simple_forward,
            input_types=[input_type]
        )
        
        # Check graph output method
        if hasattr(test_graph, 'output'):
            output = test_graph.output()
            print(f"   Graph output: {type(output)}")
            
            # Try loading the output
            session = InferenceSession()
            session.load(output)
            print("   ✅ Graph output loaded!")
            return session
        
    except Exception as e:
        print(f"   ❌ Output approach failed: {e}")
    
    return None

def test_semantic_search_pattern():
    """Test the semantic search pattern specifically."""
    print("\n=== Testing Semantic Search Pattern ===")
    
    try:
        # Create our actual semantic search graph
        dtype = DType.float32
        device = DeviceRef.CPU()
        
        query_type = TensorType(dtype, [1, 768], device)
        corpus_type = TensorType(dtype, [1000, 768], device)
        
        def semantic_forward(query, corpus):
            # Simplified version for testing
            corpus_t = ops.transpose(corpus, axis_1=0, axis_2=1)
            similarities = ops.matmul(query, corpus_t)
            return similarities
        
        search_graph = g.Graph(
            name="semantic_search_test",
            forward=semantic_forward,
            input_types=[query_type, corpus_type]
        )
        
        print("✅ Semantic search graph created")
        
        # Try the working pattern
        session = InferenceSession()
        session.load(search_graph)
        print("✅ Semantic search graph loaded!")
        
        # Test execution
        query = np.random.randn(1, 768).astype(np.float32)
        corpus = np.random.randn(1000, 768).astype(np.float32)
        
        # Check if session has execute method
        if hasattr(session, 'execute'):
            result = session.execute([query, corpus])
            print(f"✅ Execution successful! Result shape: {result[0].shape if hasattr(result[0], 'shape') else 'unknown'}")
            return session
        else:
            print("⚠️  Session doesn't have execute method")
            print(f"   Available methods: {[x for x in dir(session) if not x.startswith('_')]}")
        
    except Exception as e:
        print(f"❌ Semantic search pattern failed: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def investigate_session_methods():
    """Investigate what methods are available on InferenceSession."""
    print("\n=== InferenceSession Methods Investigation ===")
    
    try:
        session = InferenceSession()
        methods = [x for x in dir(session) if not x.startswith('_')]
        
        print("Available methods:")
        for method in sorted(methods):
            print(f"  - {method}")
        
        # Check if load method exists and get help
        if hasattr(session, 'load'):
            print(f"\nload() method help:")
            help(session.load)
        
        return session
        
    except Exception as e:
        print(f"Failed to create session for investigation: {e}")
        return None

def main():
    """Main test function."""
    print("🔧 Testing Correct MAX Graph Execution Pattern")
    print("=" * 60)
    
    # 1. Test basic load pattern
    session1 = test_inference_session_load_pattern()
    
    # 2. Test serialization approaches
    session2 = test_graph_serialization()
    
    # 3. Investigate session methods
    session3 = investigate_session_methods()
    
    # 4. Test semantic search specifically
    session4 = test_semantic_search_pattern()
    
    # Summary
    print(f"\n🎯 Results Summary")
    print("=" * 30)
    
    working_sessions = [s for s in [session1, session2, session3, session4] if s is not None]
    
    if working_sessions:
        print(f"✅ Found {len(working_sessions)} working patterns!")
        print("   Ready to implement in semantic search")
        return working_sessions[0]
    else:
        print("❌ No working execution patterns found")
        print("   Need alternative approach or API fix")
        return None

if __name__ == "__main__":
    main()