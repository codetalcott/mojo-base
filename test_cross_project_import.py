#!/usr/bin/env python3
"""
Test Cross-Project Import Compatibility
Validates that the package can be easily imported and used in other projects.
"""

import sys
from pathlib import Path

def test_main_imports():
    """Test importing main package components."""
    print("🧪 Testing Main Package Imports")
    print("=" * 40)
    
    try:
        # Test main package import
        from src import (
            MaxGraphConfig, 
            MaxSemanticSearchGraph,
            create_search_engine,
            MAX_GRAPH_AVAILABLE
        )
        print("✅ Main package imports: Success")
        print(f"   MAX Graph available: {MAX_GRAPH_AVAILABLE}")
        
        # Test convenience function
        if MAX_GRAPH_AVAILABLE:
            engine = create_search_engine(corpus_size=1000, device="cpu")
            print("✅ Convenience function: Success")
            print(f"   Engine type: {type(engine).__name__}")
        else:
            print("⚠️  Convenience function: Skipped (MAX not available)")
            
    except Exception as e:
        print(f"❌ Main package imports: Failed ({e})")
        return False
    
    return True

def test_module_imports():
    """Test importing specific modules."""
    print("\n🔧 Testing Module-Level Imports")
    print("=" * 40)
    
    try:
        # Test MAX Graph module
        from src.max_graph import MaxGraphConfig, MaxSemanticSearchGraph
        print("✅ MAX Graph module: Success")
        
        # Test creating config (will fail gracefully if MAX not available)
        try:
            config = MaxGraphConfig(corpus_size=2000, device="cpu")
            print(f"   Config created: {config.corpus_size} vectors, {config.device}")
            print(f"   Adaptive fusion: {config.enable_fusion}")
        except ImportError as import_err:
            print(f"   Config creation: Skipped (MAX not available: {import_err})")
        
    except Exception as e:
        print(f"❌ MAX Graph module: Failed ({e})")
        return False
    
    try:
        # Test integration module
        from src.integration import MCPOptimizedBridge, MCP_AVAILABLE
        print("✅ Integration module: Success")
        print(f"   MCP available: {MCP_AVAILABLE}")
        
        if MCP_AVAILABLE:
            # Test configurable bridge
            bridge = MCPOptimizedBridge(
                corpus_path="/custom/path/corpus.json",
                project_root="/custom/project"
            )
            print("✅ Configurable MCP bridge: Success")
        
    except Exception as e:
        print(f"❌ Integration module: Failed ({e})")
        return False
    
    return True

def test_cross_project_usage():
    """Test usage pattern for other projects."""
    print("\n🚀 Testing Cross-Project Usage Pattern")
    print("=" * 40)
    
    try:
        # Simulate usage in another project
        from src import create_search_engine, MAX_GRAPH_AVAILABLE
        
        if not MAX_GRAPH_AVAILABLE:
            print("⚠️  Cross-project test: Skipped (MAX not available)")
            return True
        
        # Create engine with custom configuration
        engine = create_search_engine(
            corpus_size=5000,
            device="cpu",
            use_fp16=False,
            enable_fusion=False  # Explicit override
        )
        
        print("✅ Engine creation: Success")
        print(f"   Corpus size: {engine.config.corpus_size}")
        print(f"   Device: {engine.config.device}")
        print(f"   FP16: {engine.config.use_fp16}")
        print(f"   Fusion: {engine.config.enable_fusion}")
        
        # Test configuration validation
        assert engine.config.corpus_size == 5000
        assert engine.config.device == "cpu"
        assert engine.config.enable_fusion == False  # Should be explicitly set
        
        print("✅ Configuration validation: Success")
        
    except Exception as e:
        print(f"❌ Cross-project usage: Failed ({e})")
        return False
    
    return True

def test_api_compatibility():
    """Test API import compatibility."""
    print("\n🌐 Testing API Import Compatibility")
    print("=" * 40)
    
    try:
        # Test API can import the package
        from src.integration import MCPOptimizedBridge
        
        # Test configurable bridge creation
        bridge = MCPOptimizedBridge(
            corpus_path="./data/test_corpus.json",
            onedev_path="./external/onedev",
            project_root="."
        )
        
        print("✅ API compatibility: Success")
        print(f"   Corpus path: {bridge.portfolio_corpus_path}")
        print(f"   Project path: {bridge.mojo_project_path}")
        
    except Exception as e:
        print(f"❌ API compatibility: Failed ({e})")
        return False
    
    return True

def main():
    """Run all cross-project import tests."""
    print("🚀 Cross-Project Import Compatibility Test")
    print("=" * 60)
    print("Testing package organization for easy reuse in other projects")
    print()
    
    # Add current directory to path for testing
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    tests = [
        test_main_imports,
        test_module_imports, 
        test_cross_project_usage,
        test_api_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: Exception ({e})")
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready for cross-project use.")
        print("\n💡 Usage in other projects:")
        print("   from mojo_semantic_search.src import create_search_engine")
        print("   engine = create_search_engine(5000, device='cpu')")
        print("   engine.compile()")
    else:
        print("⚠️  Some tests failed. Check imports and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)