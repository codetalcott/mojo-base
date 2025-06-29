#!/usr/bin/env python3
"""
Real GPU Autotuning on Lambda Cloud A10
Launch A10 in us-east-1 and run actual GPU autotuning
"""

import sys
sys.path.append('.')
from lambda_cloud_autotuning import LambdaCloudAPI, GPUAutotuningRunner
import asyncio

async def run_real_autotuning():
    """Launch A10 in us-east-1 and run real GPU autotuning."""
    
    print("🚀 Real Lambda Cloud A10 GPU Autotuning")
    print("=" * 50)
    
    try:
        # Override region to us-east-1 where A10 is available
        runner = GPUAutotuningRunner()
        
        # Force use of us-east-1 region and A10
        def patched_launch_instance(instance_type="gpu_1x_a10", name="mojo-autotuning"):
            # Get SSH key
            ssh_keys = runner.lambda_api.list_ssh_keys()
            ssh_key_name = ssh_keys['data'][0]['name']  # Use hackathon_lambda
            
            data = {
                "region_name": "us-east-1",  # Force us-east-1
                "instance_type_name": "gpu_1x_a10",  # Force A10
                "ssh_key_names": [ssh_key_name],
                "name": name
            }
            
            import requests
            response = requests.post(f"{runner.lambda_api.base_url}/instance-operations/launch", 
                                   headers=runner.lambda_api.headers, json=data)
            
            if response.status_code != 200:
                print(f"❌ Launch failed: {response.text}")
                response.raise_for_status()
                
            print("✅ A10 GPU launched in us-east-1")
            return response.json()
        
        # Patch the launch method
        runner.lambda_api.launch_instance = patched_launch_instance
        
        # Run the complete autotuning workflow
        print("🔥 Starting real GPU autotuning on Lambda Cloud A10...")
        result = await runner.run_complete_autotuning()
        
        if result['success']:
            print(f"\n🎉 REAL GPU AUTOTUNING COMPLETE!")
            print(f"   Instance: {result['instance_ip']}")
            print(f"   Results: {result['results_file']}")
            
            # Show key metrics
            results = result['results']
            print(f"\n📊 Real Hardware Results:")
            print(f"   ⚡ Latency: {results['best_config']['latency_ms']:.2f}ms")
            print(f"   🚀 Improvement: {results['improvement_factor']:.1f}x")
            print(f"   🧬 Corpus: {results['corpus_size']:,} vectors")
            print(f"   💾 GPU: {results['gpu_type']}")
            
        else:
            print(f"❌ Autotuning failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_real_autotuning())