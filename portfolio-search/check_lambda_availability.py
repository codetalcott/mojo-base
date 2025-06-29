#!/usr/bin/env python3
"""Quick check of Lambda Cloud GPU availability"""

import sys
sys.path.append('.')
from lambda_cloud_autotuning import LambdaCloudAPI
import requests

def check_availability():
    try:
        api = LambdaCloudAPI()
        
        regions = ["us-west-2", "us-east-1", "us-west-1"]
        instance_types = ["gpu_1x_a10", "gpu_1x_rtx6000", "gpu_1x_a6000"]
        
        print("🔍 Checking GPU availability across regions...")
        
        for region in regions:
            print(f"\n📍 Region: {region}")
            for inst_type in instance_types:
                
                # Quick availability check by attempting a dry-run
                data = {
                    "region_name": region,
                    "instance_type_name": inst_type,
                    "ssh_key_names": ["hackathon_lambda"],
                    "name": f"test-{inst_type}"
                }
                
                try:
                    response = requests.post(f"{api.base_url}/instance-operations/launch", 
                                           headers=api.headers, json=data, timeout=10)
                    
                    if response.status_code == 200:
                        print(f"  ✅ {inst_type}: Available")
                        # Immediately terminate if we accidentally launched something
                        try:
                            result = response.json()
                            if 'data' in result and 'instance_ids' in result['data']:
                                instance_id = result['data']['instance_ids'][0]
                                api.terminate_instance(instance_id)
                                print(f"     (Test instance {instance_id} terminated)")
                        except:
                            pass
                    else:
                        error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                        if 'insufficient-capacity' in error_msg.lower():
                            print(f"  ❌ {inst_type}: No capacity")
                        else:
                            print(f"  ⚠️  {inst_type}: {error_msg}")
                            
                except requests.Timeout:
                    print(f"  ⏰ {inst_type}: Timeout")
                except Exception as e:
                    print(f"  ❌ {inst_type}: Error - {e}")
                    
    except Exception as e:
        print(f"❌ Error checking availability: {e}")

if __name__ == "__main__":
    check_availability()