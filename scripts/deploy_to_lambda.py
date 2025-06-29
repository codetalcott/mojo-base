#!/usr/bin/env python3
"""
Deploy to Lambda Cloud for Real GPU Autotuning
Set up and deploy the Mojo semantic search system to Lambda Cloud A10 GPUs
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests

class LambdaCloudDeployer:
    """Deploy Mojo semantic search to Lambda Cloud for real GPU autotuning."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.instance_name = "mojo-autotuning"
        self.instance_type = "gpu_1x_a10"  # A10 GPU
        self.region = "us-east-1"
        
    def check_lambda_cli(self) -> bool:
        """Check if Lambda Cloud CLI is installed and configured."""
        try:
            result = subprocess.run(['lambda', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Lambda CLI found: {result.stdout.strip()}")
                return True
            else:
                print("❌ Lambda CLI not working properly")
                return False
        except FileNotFoundError:
            print("❌ Lambda CLI not found")
            return False
            
    def check_lambda_auth(self) -> bool:
        """Check if user is authenticated with Lambda Cloud."""
        try:
            result = subprocess.run(['lambda', 'cloud', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Lambda Cloud authentication verified")
                return True
            else:
                print("❌ Lambda Cloud authentication failed")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error checking Lambda auth: {e}")
            return False
            
    def prepare_deployment_package(self) -> bool:
        """Prepare the deployment package for Lambda Cloud."""
        print("📦 Preparing deployment package...")
        
        # Check required files
        required_files = [
            "data/real_vector_corpus.json",
            "scripts/start_autotuning.py",
            "lambda_autotuning/gpu_autotuning_lambda.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
            
        # Check corpus size
        corpus_file = self.project_root / "data/real_vector_corpus.json"
        try:
            with open(corpus_file, 'r') as f:
                corpus_data = json.load(f)
                vector_count = len(corpus_data.get('vectors', []))
                
            print(f"✅ Corpus ready: {vector_count:,} vectors")
            
            if vector_count < 1000:
                print(f"⚠️  Small corpus size: {vector_count}")
                
        except Exception as e:
            print(f"❌ Error reading corpus: {e}")
            return False
            
        return True
        
    def create_lambda_instance(self) -> Optional[str]:
        """Create a Lambda Cloud instance for autotuning."""
        print(f"🚀 Creating Lambda Cloud instance: {self.instance_name}")
        
        try:
            # Launch instance
            cmd = [
                'lambda', 'cloud', 'instances', 'launch',
                '--instance-type', self.instance_type,
                '--name', self.instance_name,
                '--region', self.region,
                '--file-systems', 'none'  # No persistent storage needed
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Instance creation initiated")
                print(result.stdout)
                
                # Extract instance ID from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Instance ID:' in line:
                        instance_id = line.split(':')[1].strip()
                        return instance_id
                        
                return "created"  # Fallback if ID not found
            else:
                print(f"❌ Instance creation failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating instance: {e}")
            return None
            
    def wait_for_instance_ready(self, instance_id: str) -> bool:
        """Wait for instance to be ready."""
        print(f"⏳ Waiting for instance {instance_id} to be ready...")
        
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                result = subprocess.run([
                    'lambda', 'cloud', 'instances', 'list'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Parse output to check instance status
                    if 'running' in result.stdout.lower():
                        print("✅ Instance is running")
                        return True
                        
                print("   Still waiting...")
                time.sleep(10)
                
            except Exception as e:
                print(f"   Error checking status: {e}")
                time.sleep(10)
                
        print("❌ Timeout waiting for instance")
        return False
        
    def upload_files(self, instance_id: str) -> bool:
        """Upload necessary files to Lambda instance."""
        print("📤 Uploading files to Lambda instance...")
        
        # Files to upload
        upload_files = [
            "data/real_vector_corpus.json",
            "scripts/start_autotuning.py",
            "lambda_autotuning/gpu_autotuning_lambda.py",
            "autotuning_results/"
        ]
        
        try:
            for file_path in upload_files:
                src_path = self.project_root / file_path
                if src_path.exists():
                    # Use scp-like command for Lambda
                    cmd = [
                        'lambda', 'cloud', 'instances', 'push',
                        instance_id, str(src_path), f"/tmp/{file_path}"
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"   ✅ Uploaded: {file_path}")
                    else:
                        print(f"   ❌ Failed to upload: {file_path}")
                        print(f"      Error: {result.stderr}")
                else:
                    print(f"   ⚠️  File not found: {file_path}")
                    
            return True
            
        except Exception as e:
            print(f"❌ Error uploading files: {e}")
            return False
            
    def run_autotuning_on_gpu(self, instance_id: str) -> bool:
        """Run the actual GPU autotuning on Lambda Cloud."""
        print("🔥 Starting real GPU autotuning on Lambda Cloud...")
        
        # Commands to run on the instance
        setup_commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-pip",
            "pip3 install numpy asyncio pathlib",
            "cd /tmp && ls -la",
            "python3 --version",
            "nvidia-smi"  # Check GPU availability
        ]
        
        autotuning_command = "cd /tmp && python3 scripts/start_autotuning.py --gpu-mode"
        
        try:
            # Run setup commands
            for cmd in setup_commands:
                print(f"   Running: {cmd}")
                result = subprocess.run([
                    'lambda', 'cloud', 'instances', 'exec', 
                    instance_id, '--', 'bash', '-c', cmd
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"   ✅ Success")
                    if result.stdout.strip():
                        print(f"      Output: {result.stdout.strip()}")
                else:
                    print(f"   ⚠️  Command failed (may be non-critical)")
                    if result.stderr.strip():
                        print(f"      Error: {result.stderr.strip()}")
                        
            # Run autotuning
            print(f"\n🚀 Starting GPU autotuning...")
            print(f"   This will take 15-30 minutes on real A10 GPU")
            
            result = subprocess.run([
                'lambda', 'cloud', 'instances', 'exec',
                instance_id, '--', 'bash', '-c', autotuning_command
            ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                print("✅ GPU autotuning completed successfully!")
                print("Output:")
                print(result.stdout)
                return True
            else:
                print("❌ GPU autotuning failed")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Autotuning timed out (30 minutes)")
            return False
        except Exception as e:
            print(f"❌ Error running autotuning: {e}")
            return False
            
    def download_results(self, instance_id: str) -> bool:
        """Download autotuning results from Lambda instance."""
        print("📥 Downloading autotuning results...")
        
        results_dir = self.project_root / "lambda_autotuning_results"
        results_dir.mkdir(exist_ok=True)
        
        try:
            # Download results files
            result_files = [
                "/tmp/autotuning_results/",
                "/tmp/optimized_kernel.mojo",
                "/tmp/autotuning_log.txt"
            ]
            
            for remote_path in result_files:
                local_path = results_dir / Path(remote_path).name
                
                cmd = [
                    'lambda', 'cloud', 'instances', 'pull',
                    instance_id, remote_path, str(local_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ Downloaded: {Path(remote_path).name}")
                else:
                    print(f"   ⚠️  Could not download: {remote_path}")
                    
            return True
            
        except Exception as e:
            print(f"❌ Error downloading results: {e}")
            return False
            
    def cleanup_instance(self, instance_id: str) -> bool:
        """Terminate the Lambda Cloud instance."""
        print(f"🧹 Cleaning up Lambda instance: {instance_id}")
        
        try:
            result = subprocess.run([
                'lambda', 'cloud', 'instances', 'terminate', instance_id
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Instance terminated")
                return True
            else:
                print(f"❌ Failed to terminate instance: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error terminating instance: {e}")
            return False
            
    async def deploy_and_run_autotuning(self) -> Dict[str, any]:
        """Complete deployment and autotuning workflow."""
        print("🚀 Lambda Cloud Real GPU Autotuning")
        print("=" * 50)
        
        # Step 1: Prerequisites
        if not self.check_lambda_cli():
            return {'success': False, 'error': 'Lambda CLI not found'}
            
        if not self.check_lambda_auth():
            return {'success': False, 'error': 'Lambda Cloud authentication failed'}
            
        if not self.prepare_deployment_package():
            return {'success': False, 'error': 'Deployment package preparation failed'}
            
        # Step 2: Create instance
        instance_id = self.create_lambda_instance()
        if not instance_id:
            return {'success': False, 'error': 'Instance creation failed'}
            
        try:
            # Step 3: Wait for instance
            if not self.wait_for_instance_ready(instance_id):
                return {'success': False, 'error': 'Instance not ready', 'instance_id': instance_id}
                
            # Step 4: Upload files
            if not self.upload_files(instance_id):
                return {'success': False, 'error': 'File upload failed', 'instance_id': instance_id}
                
            # Step 5: Run autotuning
            if not self.run_autotuning_on_gpu(instance_id):
                return {'success': False, 'error': 'GPU autotuning failed', 'instance_id': instance_id}
                
            # Step 6: Download results
            if not self.download_results(instance_id):
                return {'success': False, 'error': 'Results download failed', 'instance_id': instance_id}
                
            return {
                'success': True,
                'instance_id': instance_id,
                'message': 'Real GPU autotuning completed successfully!'
            }
            
        finally:
            # Always cleanup
            self.cleanup_instance(instance_id)

def show_lambda_setup_instructions():
    """Show instructions for Lambda Cloud setup."""
    print("🔧 Lambda Cloud Setup Instructions")
    print("=" * 40)
    print()
    print("1. Install Lambda Cloud CLI:")
    print("   pip install lambda-cloud")
    print()
    print("2. Create Lambda Cloud account:")
    print("   https://lambdalabs.com/service/gpu-cloud")
    print()
    print("3. Get API key and authenticate:")
    print("   lambda cloud auth")
    print()
    print("4. Verify setup:")
    print("   lambda cloud instances list")
    print()
    print("Once setup is complete, run this script again!")

async def main():
    """Main deployment function."""
    deployer = LambdaCloudDeployer()
    
    # Check if Lambda CLI is available
    if not deployer.check_lambda_cli():
        show_lambda_setup_instructions()
        return
        
    # Run deployment
    result = await deployer.deploy_and_run_autotuning()
    
    if result['success']:
        print(f"\n🎉 Real GPU Autotuning Complete!")
        print(f"   Results available in: lambda_autotuning_results/")
        print(f"   Instance ID: {result['instance_id']}")
    else:
        print(f"\n❌ Deployment failed: {result['error']}")
        if 'instance_id' in result:
            print(f"   Instance ID: {result['instance_id']} (may need manual cleanup)")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())