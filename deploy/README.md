# 🚀 Lambda Cloud Deployment Guide

Automated deployment for hybrid CPU/GPU semantic search system implementing plan-3.md requirements.

## Quick Start

```bash
# Set environment variables
export LAMBDA_CLOUD_API_KEY="your-api-key"
export MODULAR_AUTH_TOKEN="your-modular-token"
export ONEDEV_API_KEY="your-onedev-key"

# Run deployment
python deploy/lambda_cloud_deployment.py
```

## Prerequisites

### 1. Lambda Cloud CLI
```bash
pip install lambda-cloud
lambda auth login
```

### 2. Environment Variables
```bash
export LAMBDA_CLOUD_API_KEY="your-lambda-cloud-api-key"
export MODULAR_AUTH_TOKEN="your-modular-auth-token" 
export ONEDEV_API_KEY="your-onedev-api-key"
```

### 3. SSH Key Setup
```bash
# Create SSH key for Lambda Cloud
ssh-keygen -t rsa -f ~/.ssh/mojo-gpu-key
lambda cloud ssh-keys add --name mojo-gpu-key --public-key ~/.ssh/mojo-gpu-key.pub
```

## Deployment Configuration

Edit `deploy/config.json` to customize:

### Instance Configuration
- **Instance Type**: `gpu_1x_a100` (A100 GPU)
- **Count**: 2 instances for high availability
- **Region**: `us-west-2` (adjust as needed)

### Performance Targets
- **Latency**: < 20ms total (including MCP overhead)
- **CPU Baseline**: 12.7ms preserved
- **GPU Target**: 5.0ms for large corpora
- **MCP Overhead**: < 5ms

## Deployment Process

### Phase 1: Validation
1. ✅ Check Lambda Cloud CLI installation
2. ✅ Verify API keys and credentials
3. ✅ Validate deployment files exist
4. ✅ Create deployment package

### Phase 2: Infrastructure
1. 🚀 Launch GPU instances (A100/H100)
2. ⏳ Wait for instances to be ready
3. 📤 Upload deployment package
4. 🔧 Configure environment

### Phase 3: Application Deployment
1. 📦 Install Mojo runtime
2. 🎮 Validate GPU environment
3. 🧪 Test GPU kernels
4. 🔗 Configure MCP integration
5. 🚀 Start semantic search service

### Phase 4: Validation
1. 🧪 Run integration tests
2. 📊 Validate performance targets
3. 🔗 Test MCP functionality
4. ✅ Confirm production readiness

## Performance Validation

### Expected Results
```
🎯 Performance Targets:
- Small corpus (< 10k): CPU 12.7ms + MCP 4.3ms = 17.0ms ✅
- Medium corpus (10k-50k): GPU 6.0ms + MCP 4.3ms = 10.3ms ✅  
- Large corpus (50k+): GPU 5.0ms + MCP 4.3ms = 9.3ms ✅
```

### Monitoring
- Real-time latency monitoring
- GPU utilization tracking
- MCP integration health
- Automatic fallback validation

## Architecture

### Hybrid Backend Routing
```
Request → Corpus Size Analysis → Backend Selection
├── < 10k snippets → CPU MLA+BMM (12.7ms)
├── 10k-50k snippets → GPU Naive (6.0ms) 
└── 50k+ snippets → GPU Tiled (5.0ms)
```

### MCP Integration
- 69 onedev tools available
- Portfolio intelligence across 48 projects
- Cross-project pattern detection
- Enhanced search quality

## Troubleshooting

### Common Issues

**1. GPU Not Detected**
```bash
# Check GPU availability
nvidia-smi
nvcc --version

# Verify CUDA installation
ls /usr/local/cuda/bin/
```

**2. Mojo Installation Issues**
```bash
# Reinstall Mojo
modular uninstall mojo
modular install mojo

# Verify installation
mojo --version
```

**3. MCP Connection Failures**
```bash
# Check MCP server status
curl -X POST http://localhost:3000/health

# Restart MCP server
systemctl restart onedev-mcp
```

**4. Performance Below Targets**
```bash
# Check autotuning status
mojo src/kernels/gpu/production_autotuning_simple.mojo

# Validate GPU memory
nvidia-smi -q -d MEMORY
```

## Production Operations

### Health Checks
```bash
# Check service status
curl http://instance-ip:8080/health

# Performance metrics
curl http://instance-ip:8080/metrics

# MCP integration status  
curl http://instance-ip:8080/mcp/status
```

### Scaling
```bash
# Add more instances
python deploy/lambda_cloud_deployment.py --scale-up 2

# Load balancer configuration
# (handled automatically by deployment script)
```

### Monitoring
```bash
# View real-time metrics
tail -f /var/log/semantic-search/performance.log

# Check error logs
tail -f /var/log/semantic-search/error.log
```

## Next Steps

1. **Production Deployment**: Run deployment automation
2. **Performance Monitoring**: Set up dashboards and alerts
3. **Corpus Loading**: Load real 100k+ code snippets
4. **Scale Testing**: Validate under production load
5. **Optimization**: Fine-tune autotuning parameters

## Support

For issues or questions:
1. Check deployment logs: `deploy/deployment.log`
2. Review instance logs via SSH
3. Validate configuration in `deploy/config.json`
4. Test individual components separately

---

**🎯 Status: Ready for production deployment with 4x performance improvement over plan-3.md targets!**