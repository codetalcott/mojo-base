#!/usr/bin/env python3
import sys
sys.path.append('.')
from lambda_cloud_autotuning import LambdaCloudAPI

try:
    api = LambdaCloudAPI()
    ssh_keys = api.list_ssh_keys()
    print('🔑 SSH Keys Response:', ssh_keys)
    
    if 'data' in ssh_keys and ssh_keys['data']:
        print('✅ Available SSH keys:')
        for key in ssh_keys['data']:
            print(f'  - {key}')
    else:
        print('❌ No SSH keys found')
        
except Exception as e:
    print(f'❌ Error: {e}')