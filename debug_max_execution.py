#!/usr/bin/env python3
"""
Debug MAX Graph execution issues systematically
"""

import max.graph as g
from max.graph import ops, TensorType, DeviceRef
from max.dtype import DType
from max.engine import InferenceSession, Model
import numpy as np

def investigate_inference_session():
    """Investigate InferenceSession requirements and alternatives."""
    print("=== InferenceSession Investigation ===")
    
    # Check InferenceSession constructor
    print("InferenceSession help:")
    help(InferenceSession.__init__)
    print()
    
    # Check if there are alternative constructors or methods
    print("InferenceSession methods:")
    methods = [x for x in dir(InferenceSession) if not x.startswith('_')]
    for method in sorted(methods):
        print(f"  - {method}")
    print()

def investigate_model_conversion():
    """Check if Graph needs to be converted to Model first."""
    print("=== Model Conversion Investigation ===")
    
    print("Model help:")
    help(Model.__init__)
    print()
    
    print("Model methods:")
    methods = [x for x in dir(Model) if not x.startswith('_')]
    for method in sorted(methods):
        print(f"  - {method}")
    print()

def test_alternative_execution_patterns():
    """Test different execution patterns."""
    print("=== Testing Alternative Execution Patterns ===")
    
    # Create a simple test graph
    dtype = DType.float32
    device = DeviceRef.CPU()
    input_type = TensorType(dtype, [2, 3], device)
    
    def simple_forward(x):
        return ops.add(x, x)
    
    test_graph = g.Graph(
        name="test_execution",
        forward=simple_forward,
        input_types=[input_type]
    )
    
    print("✅ Test graph created")
    
    # Test 1: Direct InferenceSession with different parameters
    print("\nTest 1: InferenceSession variations...")
    
    # Try with no parameters
    try:
        session1 = InferenceSession(test_graph)
        print("✅ InferenceSession() - basic constructor works")
        return session1
    except Exception as e:
        print(f"❌ InferenceSession() failed: {e}")
    
    # Try saving graph first
    print("\nTest 2: Saving graph first...")
    try:
        import tempfile
        import os
        
        # Save graph to temporary file
        with tempfile.NamedTemporaryFile(suffix='.maxgraph', delete=False) as f:
            temp_path = f.name
        
        # Check if graph has save method
        if hasattr(test_graph, 'save'):
            test_graph.save(temp_path)
            print(f"✅ Graph saved to {temp_path}")
            
            # Try loading with InferenceSession
            session2 = InferenceSession(temp_path)
            print("✅ InferenceSession from file works")
            
            # Cleanup
            os.unlink(temp_path)
            return session2
        else:
            print("❌ Graph doesn't have save method")
            
    except Exception as e:
        print(f"❌ Save/load approach failed: {e}")
    
    # Test 3: Check if we need to use Model instead
    print("\nTest 3: Model-based approach...")
    try:
        # Try creating a Model from the graph
        model = Model(test_graph)
        print("✅ Model created from graph")
        
        # Try InferenceSession with Model
        session3 = InferenceSession(model)
        print("✅ InferenceSession with Model works")
        return session3
        
    except Exception as e:
        print(f"❌ Model approach failed: {e}")
    
    # Test 4: Check graph compilation state
    print("\nTest 4: Graph state investigation...")
    print(f"Graph methods: {[x for x in dir(test_graph) if not x.startswith('_')]}")
    
    # Check if graph needs compilation step
    if hasattr(test_graph, 'compile'):
        try:
            test_graph.compile()
            print("✅ Graph compilation step completed")
            
            session4 = InferenceSession(test_graph)
            print("✅ InferenceSession after compilation works")
            return session4
            
        except Exception as e:
            print(f"❌ Compilation approach failed: {e}")
    
    return None

def investigate_engine_alternatives():
    """Look for alternative execution engines."""
    print("\n=== Engine Alternatives Investigation ===")
    
    # Check what's available in max.engine
    import max.engine as engine
    
    print("max.engine contents:")
    engine_contents = [x for x in dir(engine) if not x.startswith('_')]
    for item in sorted(engine_contents):
        print(f"  - {item}")
    
    # Check for alternative execution methods
    execution_candidates = [x for x in engine_contents if 'exec' in x.lower() or 'run' in x.lower() or 'session' in x.lower()]
    print(f"\nExecution candidates: {execution_candidates}")

def test_graph_direct_execution():
    """Test if graphs can be executed directly without InferenceSession."""
    print("\n=== Direct Graph Execution Test ===")
    
    # Create simple graph
    dtype = DType.float32
    device = DeviceRef.CPU()
    input_type = TensorType(dtype, [2, 3], device)
    
    def simple_forward(x):
        return ops.add(x, x)
    
    test_graph = g.Graph(
        name="direct_test",
        forward=simple_forward,
        input_types=[input_type]
    )
    
    # Check if graph is callable
    if callable(test_graph):
        try:
            test_input = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            result = test_graph(test_input)
            print("✅ Direct graph execution works")
            print(f"   Result: {result}")
            return True
        except Exception as e:
            print(f"❌ Direct execution failed: {e}")
    else:
        print("❌ Graph is not callable")
    
    # Check for execute/run methods on graph
    graph_methods = [x for x in dir(test_graph) if 'exec' in x.lower() or 'run' in x.lower() or 'call' in x.lower()]
    print(f"Graph execution methods: {graph_methods}")
    
    return False

def investigate_mlir_context():
    """Check if MLIR context setup is needed."""
    print("\n=== MLIR Context Investigation ===")
    
    try:
        # Check if there's mlir module available
        import max._core as core
        print("max._core available")
        
        # Look for context-related functionality
        core_contents = [x for x in dir(core) if not x.startswith('_')]
        context_items = [x for x in core_contents if 'context' in x.lower() or 'mlir' in x.lower()]
        print(f"Context-related items: {context_items}")
        
    except ImportError:
        print("max._core not available")
    
    # Check graph constructor parameters again
    print("\nGraph constructor parameters:")
    help(g.Graph.__init__)

def main():
    """Main investigation function."""
    print("🔍 MAX Graph Execution Issue Investigation")
    print("=" * 60)
    
    # 1. Investigate InferenceSession requirements
    investigate_inference_session()
    
    # 2. Check Model conversion approach
    investigate_model_conversion()
    
    # 3. Test alternative execution patterns
    working_session = test_alternative_execution_patterns()
    
    # 4. Look for engine alternatives
    investigate_engine_alternatives()
    
    # 5. Test direct graph execution
    direct_works = test_graph_direct_execution()
    
    # 6. Investigate MLIR context requirements
    investigate_mlir_context()
    
    # Summary
    print(f"\n🎯 Investigation Summary")
    print("=" * 30)
    
    if working_session:
        print("✅ Found working execution pattern!")
        print("   Can proceed with MAX Graph execution")
    elif direct_works:
        print("✅ Direct graph execution works!")
        print("   Can bypass InferenceSession")
    else:
        print("❌ No working execution pattern found")
        print("   Need to investigate further or use different approach")
    
    print(f"\n💡 Next Steps:")
    print("1. Try the working pattern in semantic search implementation")
    print("2. Look for MAX Graph examples in installation")
    print("3. Check MAX documentation for execution patterns")
    print("4. Consider alternative approaches (direct execution, etc.)")

if __name__ == "__main__":
    main()