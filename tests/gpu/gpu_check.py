#!/usr/bin/env python3
"""
Simple GPU check and performance test
"""

import subprocess
import json
import time
import sys

def check_gpu():
    """Check GPU availability."""
    print("🔍 Checking GPU...")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu"], 
            capture_output=True, text=True, check=True
        )
        print(f"✅ GPU detected: {result.stdout.strip()}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ No GPU detected or nvidia-smi not available")
        return False

def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy"], check=True)
        print("✅ numpy installed")
    except subprocess.SubprocessError:
        print("❌ Failed to install numpy")
        return False
    return True

def simple_performance_test():
    """Run simple performance test."""
    print("🚀 Running simple performance test...")
    
    try:
        import numpy as np
    except ImportError:
        print("Installing numpy...")
        if not install_dependencies():
            return
        import numpy as np
    
    # Test configurations
    configs = [
        {"tile": 4, "block": 32},
        {"tile": 8, "block": 32}, 
        {"tile": 16, "block": 32},
        {"tile": 32, "block": 32}
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/4] Testing tile={config['tile']}, block={config['block']}")
        
        # Matrix size based on config
        size = config['tile'] * config['block']
        
        # Create random matrices
        a = np.random.random((size, 128)).astype(np.float32)
        b = np.random.random((128, size)).astype(np.float32)
        
        # Time the operation
        start_time = time.time()
        c = np.dot(a, b)
        latency = (time.time() - start_time) * 1000
        
        result = {
            "config": config,
            "latency_ms": round(latency, 2),
            "matrix_size": size
        }
        results.append(result)
        
        print(f"   Latency: {latency:.2f}ms")
    
    # Find best result
    best = min(results, key=lambda x: x["latency_ms"])
    
    final_results = {
        "gpu": "local",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "results": results,
        "best": best
    }
    
    # Save results
    with open("gpu_test_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n🎉 Test complete!")
    print(f"Best: tile={best['config']['tile']}, latency={best['latency_ms']}ms")
    print("Results saved to gpu_test_results.json")

if __name__ == "__main__":
    print("🔧 GPU Performance Test")
    print("=" * 25)
    
    # Check for GPU
    has_gpu = check_gpu()
    
    # Run performance test
    simple_performance_test()
