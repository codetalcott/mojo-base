#!/usr/bin/env python3
"""
Test future-proof adaptive fusion detection.
Validates that Apple Metal and other GPU architectures are automatically detected.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from max_graph.semantic_search_graph import MaxGraphConfig

def test_device_detection():
    """Test various device types and their fusion settings."""
    
    test_cases = [
        # Current devices
        ("cpu", False, "CPU should disable fusion"),
        ("gpu", True, "Generic GPU should enable fusion"),
        ("cuda", True, "NVIDIA CUDA should enable fusion"),
        
        # Future Apple Metal support
        ("metal", True, "Apple Metal should enable fusion"),
        ("apple_metal", True, "Apple Metal variant should enable fusion"),
        ("mlx", True, "Apple MLX should enable fusion"),
        
        # Other GPU architectures
        ("rocm", True, "AMD ROCm should enable fusion"),
        ("opencl", True, "OpenCL should enable fusion"),
        ("vulkan", True, "Vulkan should enable fusion"),
        ("tensorrt", True, "TensorRT should enable fusion"),
        ("dml", True, "DirectML should enable fusion"),
        
        # Mixed case testing
        ("GPU", True, "Uppercase GPU should work"),
        ("Metal", True, "Capitalized Metal should work"),
        ("CUDA", True, "Uppercase CUDA should work"),
        
        # Non-parallel devices
        ("cpu_only", False, "CPU-only should disable fusion"),
        ("arm_cpu", False, "ARM CPU should disable fusion"),
        ("x86", False, "x86 should disable fusion"),
    ]
    
    print("🧪 Testing Future-Proof Device Detection")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for device, expected_fusion, description in test_cases:
        config = MaxGraphConfig(
            corpus_size=1000,
            device=device
        )
        
        actual_fusion = config.enable_fusion
        status = "✅ PASS" if actual_fusion == expected_fusion else "❌ FAIL"
        
        print(f"{status} {device:12} → fusion={actual_fusion:5} | {description}")
        
        if actual_fusion == expected_fusion:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed! Future-proof detection working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check device detection logic.")
        return False

def test_apple_metal_readiness():
    """Test specific Apple Metal readiness scenarios."""
    
    print("\n🍎 Testing Apple Metal Readiness")
    print("=" * 40)
    
    # Test various ways Apple Metal might be specified
    apple_scenarios = [
        "metal",
        "apple_metal", 
        "Metal",
        "METAL",
        "gpu_metal",
        "metal_gpu",
        "mlx",
        "MLX"
    ]
    
    for device in apple_scenarios:
        config = MaxGraphConfig(
            corpus_size=5000,
            device=device
        )
        
        if config.enable_fusion:
            print(f"✅ {device:12} → fusion enabled (ready for Apple Metal)")
        else:
            print(f"❌ {device:12} → fusion disabled (NOT ready for Apple Metal)")
    
    print("\n📝 When Modular adds Apple Metal support:")
    print("   - No code changes needed")
    print("   - Fusion will automatically enable for Metal devices")
    print("   - Performance optimization will be immediate")

def test_explicit_override():
    """Test that explicit fusion setting still works."""
    
    print("\n🎛️  Testing Explicit Override")
    print("=" * 35)
    
    # Test explicit True override
    config_force_true = MaxGraphConfig(
        corpus_size=1000,
        device="cpu",
        enable_fusion=True  # Force fusion on CPU
    )
    
    # Test explicit False override
    config_force_false = MaxGraphConfig(
        corpus_size=1000,
        device="gpu",
        enable_fusion=False  # Force no fusion on GPU
    )
    
    # Test auto-detection
    config_auto = MaxGraphConfig(
        corpus_size=1000,
        device="gpu"
        # enable_fusion=None (default)
    )
    
    print(f"CPU with forced fusion=True:  {config_force_true.enable_fusion}")
    print(f"GPU with forced fusion=False: {config_force_false.enable_fusion}")
    print(f"GPU with auto-detection:      {config_auto.enable_fusion}")
    
    if (config_force_true.enable_fusion == True and 
        config_force_false.enable_fusion == False and
        config_auto.enable_fusion == True):
        print("✅ Explicit override working correctly")
    else:
        print("❌ Explicit override not working")

if __name__ == "__main__":
    print("🚀 Future-Proof Fusion Detection Tests")
    print("=" * 60)
    
    success = test_device_detection()
    test_apple_metal_readiness()
    test_explicit_override()
    
    if success:
        print("\n🎉 All tests passed! Ready for Apple Metal and future GPU architectures.")
    else:
        print("\n❌ Some tests failed. Check implementation.")