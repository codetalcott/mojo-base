# Legacy Mojo Kernel Implementations

## Purpose
This directory preserves our original manual Mojo kernel implementations as a fallback option while we experiment with MAX Graph API optimizations.

## Files Preserved
- `bmm_kernel_optimized.mojo` - Original optimized BMM kernel with manual SIMD, tiling, and parallelization
- `mla_kernel_optimized.mojo` - Original optimized MLA kernel with multi-head attention
- `semantic_search_mvp.mojo` - Original semantic search MVP implementation

## Why Keep These?
1. **Fallback Safety**: If MAX Graph API doesn't perform as expected
2. **Performance Baseline**: For comparing MAX Graph optimizations
3. **Learning Reference**: Shows manual optimization techniques
4. **Autotuning Compatibility**: Works with our existing autotuning framework

## Performance Characteristics
- **Proven Performance**: These implementations have been tested and validated
- **Manual Optimizations**: Hand-tuned SIMD, cache tiling, memory management
- **Autotuning Ready**: Compatible with our autotuning V2 framework

## When to Use Legacy Implementation
- MAX Graph API performance is insufficient
- Need specific low-level optimizations
- Debugging MAX Graph issues
- Comparing optimization approaches

## Migration Status
- ✅ Preserved original implementations
- 🔄 Testing MAX Graph API alternatives
- 📊 Performance comparison in progress

The goal is to leverage MAX's automatic optimizations while maintaining our proven manual optimizations as backup.