#!/usr/bin/env python3
"""
Start autotuning on GPU 1 while waiting for GPU 2
Single GPU autotuning focused on tile optimization
"""

import asyncio
import sys
sys.path.append('.')
from dual_gpu_autotuning import DualGPUAutotuningManager, GPUInstance, AutotuningConfig

async def start_gpu1_autotuning():
    """Start autotuning on GPU 1 immediately."""
    
    print("🚀 Starting GPU 1 Autotuning (Tile Optimization)")
    print("=" * 55)
    
    # GPU 1 instance
    gpu1 = GPUInstance(
        name="mojo-autotuning-gpu1",
        ip="129.213.131.99",
        instance_id="gpu1",
        gpu_type="A10"
    )
    
    # GPU 1 configuration (tile optimization)
    config1 = AutotuningConfig(
        gpu_id=1,
        tile_sizes=[4, 8, 16, 32],
        block_sizes=[32],  # Fixed optimal block size
        shared_memory_sizes=[8192],  # Fixed optimal memory
        test_name="tile_optimization"
    )
    
    # Create manager and run GPU 1 autotuning
    manager = DualGPUAutotuningManager()
    
    print(f"🔥 Connecting to GPU 1: {gpu1.ip}")
    print("   Testing tile sizes: 4, 8, 16, 32")
    print("   Fixed block size: 32")
    print("   Expected duration: ~5-8 minutes")
    print()
    
    # Run autotuning on GPU 1
    result = await manager.upload_and_run_gpu(gpu1, config1)
    
    if result:
        print("\n🎉 GPU 1 Autotuning Complete!")
        print(f"   Best latency: {result['summary']['best_latency_ms']:.2f}ms")
        print(f"   Best tile size: {result['best_config']['config']['tile_size']}")
        print(f"   GFLOPS: {result['best_config']['performance']['gflops']:.1f}")
        print(f"   Target <10ms: {'✅' if result['summary']['best_latency_ms'] < 10.0 else '❌'}")
        
        # Save GPU 1 results
        import json
        import time
        results_file = f"gpu1_tile_optimization_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"   💾 Results saved: {results_file}")
        
        return result
    else:
        print("❌ GPU 1 autotuning failed")
        return None

if __name__ == "__main__":
    result = asyncio.run(start_gpu1_autotuning())
    
    if result:
        print("\n📋 Ready for GPU 2!")
        print("Once GPU 2 is running, provide its IP to run parallel optimization.")