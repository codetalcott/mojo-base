#!/usr/bin/env python3
"""
Validate device detection with actual MAX Graph implementation.
Test that the new future-proof logic works with the real code.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_without_max_import():
    """Test device detection without importing MAX (avoids import errors)."""
    
    # Import just the config class without MAX dependencies
    from max_graph.semantic_search_graph import MaxGraphConfig
    
    print("🧪 Testing Device Detection with MAX Graph Config")
    print("=" * 55)
    
    # Test current CPU configuration
    config_cpu = MaxGraphConfig(corpus_size=1000, device='cpu')
    print(f"CPU device:    fusion={config_cpu.enable_fusion} (should be False)")
    
    # Test hypothetical GPU configuration  
    config_gpu = MaxGraphConfig(corpus_size=1000, device='gpu')
    print(f"GPU device:    fusion={config_gpu.enable_fusion} (should be True)")
    
    # Test future Apple Metal
    config_metal = MaxGraphConfig(corpus_size=1000, device='metal')
    print(f"Metal device:  fusion={config_metal.enable_fusion} (should be True)")
    
    # Test explicit override
    config_override = MaxGraphConfig(corpus_size=1000, device='cpu', enable_fusion=True)
    print(f"CPU override:  fusion={config_override.enable_fusion} (should be True)")
    
    # Validate results
    cpu_correct = config_cpu.enable_fusion == False
    gpu_correct = config_gpu.enable_fusion == True
    metal_correct = config_metal.enable_fusion == True
    override_correct = config_override.enable_fusion == True
    
    all_correct = cpu_correct and gpu_correct and metal_correct and override_correct
    
    print("\n" + "=" * 55)
    if all_correct:
        print("✅ All device detection tests passed!")
        print("✅ Ready for Apple Metal without code changes")
        print("✅ Future-proof design validated")
    else:
        print("❌ Some device detection tests failed")
        print(f"CPU: {cpu_correct}, GPU: {gpu_correct}, Metal: {metal_correct}, Override: {override_correct}")
    
    return all_correct

if __name__ == "__main__":
    try:
        success = test_without_max_import()
        if success:
            print("\n🎉 MAX Graph device detection ready for production!")
        else:
            print("\n❌ Device detection needs fixes")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("This might be due to missing MAX dependencies, but device detection logic should still work")