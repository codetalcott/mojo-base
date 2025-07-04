# GPU Testing Guide for Modular Platform

This guide walks through testing MAX Graph GPU performance on the Modular Platform.

## Prerequisites

### 1. Access to Modular Platform
- Sign up at [modular.com](https://modular.com)
- Request GPU instance access
- Ensure you have credits for GPU compute time

### 2. Environment Setup
```bash
# In your Modular Platform environment
git clone <your-repo>
cd mojo-base
pixi install
```

### 3. GPU Instance Requirements
- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **Minimum**: 4GB VRAM for basic testing
- **CUDA**: Version 11.0+ compatible

## Running GPU Tests

### Option 1: Quick Test (Recommended)
```bash
# Run the comprehensive GPU test
cd portfolio-search
pixi run python ../scripts/gpu_performance_test.py
```

### Option 2: Manual Step-by-Step

#### Step 1: Check GPU Availability
```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Check MAX can see the GPU
pixi run python -c "import max.graph; print('MAX Graph available')"
```

#### Step 2: Test Small Configuration
```bash
# Test basic GPU compilation
pixi run python -c "
import sys
sys.path.append('src')
from max_graph.semantic_search_graph import MaxGraphConfig, MaxSemanticSearchGraph

config = MaxGraphConfig(corpus_size=1000, device='gpu')
search = MaxSemanticSearchGraph(config)
search.compile()
print('GPU compilation:', 'SUCCESS' if search.model else 'FAILED')
"
```

#### Step 3: Run Performance Benchmarks
```bash
# Run the full benchmark suite
pixi run python scripts/gpu_performance_test.py
```

## Expected Results

### GPU Performance Targets
Based on our analysis and projections:

| Corpus Size | CPU Baseline | GPU Target | Expected Speedup |
|-------------|--------------|------------|------------------|
| 2,000       | 0.91ms       | 0.09ms     | 10x             |
| 5,000       | 1.81ms       | 0.18ms     | 10x             |
| 10,000      | 3.68ms       | 0.37ms     | 10x             |

### Key Metrics to Validate
1. **FP16 Performance**: Should be 2x faster than FP32
2. **Fusion Effectiveness**: Should improve GPU performance by 10-20%
3. **Memory Usage**: Should scale linearly with corpus size
4. **Compilation Time**: Should be reasonable (<30s for large graphs)

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify MAX GPU support
pixi run python -c "import max.engine; print('MAX engine available')"
```

#### 2. Compilation Failures
- **Memory Error**: Reduce corpus size or use FP16
- **Driver Error**: Update NVIDIA drivers
- **MAX Error**: Check MAX version compatibility

#### 3. Performance Issues
- **Slow Performance**: Check GPU utilization with `nvidia-smi`
- **Memory Issues**: Monitor VRAM usage
- **Thermal Throttling**: Check GPU temperature

### Performance Analysis Commands

```bash
# Monitor GPU during testing
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Profile GPU utilization
nvidia-smi dmon -s pucvmet -d 1
```

## Interpreting Results

### Success Criteria
- ✅ **Compilation**: GPU compilation succeeds without errors
- ✅ **Performance**: GPU is 5-15x faster than CPU
- ✅ **FP16 Benefit**: FP16 is 1.5-2x faster than FP32
- ✅ **Scaling**: Linear scaling with corpus size

### Analysis Questions
1. **Is GPU faster than CPU?** Compare average latencies
2. **Does FP16 help?** Compare FP16 vs FP32 results
3. **Is fusion effective?** Check if fusion improves performance
4. **How does it scale?** Plot latency vs corpus size

## Cost Optimization

### GPU Usage Tips
- **Test incrementally**: Start with small corpus sizes
- **Use FP16**: Reduces memory usage and improves speed
- **Monitor costs**: GPU time is expensive
- **Batch tests**: Run multiple configurations in one session

### Recommended Test Sequence
1. **Validation**: 1K vectors, 1 iteration (cost: minimal)
2. **Optimization**: 2K-5K vectors, 3 iterations (cost: low)
3. **Scaling**: 10K-50K vectors, 5 iterations (cost: medium)
4. **Production**: 100K+ vectors, 10 iterations (cost: high)

## Next Steps After GPU Testing

### If GPU Tests Succeed ✅
1. **Document performance gains**
2. **Update production configuration**
3. **Plan GPU deployment strategy**
4. **Monitor Apple Metal roadmap**

### If GPU Tests Fail ❌
1. **Analyze failure modes**
2. **Optimize for current hardware**
3. **Consider alternative approaches**
4. **Plan iterative improvements**

## Support

- **Modular Platform**: Check documentation and support channels
- **MAX Graph**: Review MAX Graph API documentation
- **GPU Issues**: NVIDIA developer forums and CUDA documentation

## Files Generated

After running tests, check these files:
- `data/results/gpu_performance_YYYYMMDD_HHMMSS.json`
- Console output with detailed analysis
- Error logs if compilation fails

## Cost Estimate

Approximate costs for GPU testing on Modular Platform:
- **Basic validation**: $1-5
- **Comprehensive testing**: $10-25
- **Production scaling tests**: $25-100

*Note: Actual costs depend on instance type and usage time*