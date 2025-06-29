# GPU Migration Plan: CPU to GPU-Accelerated Semantic Search

## Executive Summary

This plan outlines the systematic migration from the current CPU-based implementation to the GPU-accelerated approach proposed in plan-3.md, while preserving existing functionality and enhancing performance.

## Current vs Proposed Architecture Comparison

### Current Implementation (CPU-Based)
- **Kernels**: MLA (Multi-Head Latent Attention) and BMM (Batched Matrix Multiplication)
- **Embeddings**: 768-dimensional vectors
- **Performance**: 12.7ms total latency (8.5ms embedding, 4.2ms search)
- **Hardware**: CPU with SIMD optimization
- **Integration**: Onedev MCP tools for portfolio intelligence

### Proposed Implementation (GPU-Based)
- **Kernels**: GPU-native matmul with shared memory tiling
- **Embeddings**: 384-dimensional vectors (GTE-small model)
- **Performance Target**: < 20ms (already achieved with CPU!)
- **Hardware**: NVIDIA GPUs (A100/H100)
- **Optimization**: Progressive from naive → tiled → autotuned

## Step-by-Step Migration Plan

### Phase 1: Analysis & Preparation (2 hours)

#### Step 1.1: Performance Baseline Documentation
- [ ] Document current CPU performance metrics
  - MLA kernel: 8.5ms for 768-dim embeddings
  - BMM kernel: 4.2ms for similarity search
  - Total: 12.7ms end-to-end latency
- [ ] Create benchmark suite for fair CPU vs GPU comparison
- [ ] Document memory usage patterns

#### Step 1.2: Architectural Assessment
- [ ] Identify reusable components:
  - Data structures (CodeSnippet, SearchResult)
  - Onedev integration bridge
  - Search engine coordination logic
- [ ] Map CPU patterns to GPU equivalents:
  - SIMD → CUDA warps
  - `parallelize` → GPU grid/blocks
  - Memory tiling → Shared memory tiling

#### Step 1.3: Model Dimension Analysis
- [ ] Analyze impact of 768-dim → 384-dim change
- [ ] Create adapter layer for dimension compatibility
- [ ] Plan for dual-model support (both dimensions)

### Phase 2: GPU Environment Setup (1 hour)

#### Step 2.1: Local GPU Development
- [ ] Verify Mojo GPU support with local NVIDIA GPU (if available)
- [ ] Update pixi.toml for GPU dependencies
- [ ] Create GPU detection and fallback logic

#### Step 2.2: Cloud Preparation
- [ ] Document Lambda Cloud setup requirements
- [ ] Create deployment scripts for GPU instances
- [ ] Plan data synchronization strategy

### Phase 3: Kernel Migration (4 hours)

#### Step 3.1: Naive GPU Kernel Implementation
- [ ] Port BMM kernel to basic GPU implementation
  ```mojo
  @kernel
  fn matmul_gpu_naive(
      C: DTypePointer[float32],
      A: DTypePointer[float32], 
      B: DTypePointer[float32],
      M: Int, N: Int, K: Int
  ):
      let idx = block_idx.x * block_dim.x + thread_idx.x
      # Basic GPU matmul implementation
  ```
- [ ] Implement GPU memory management wrapper
- [ ] Create Python-Mojo bridge for GPU operations

#### Step 3.2: Shared Memory Tiling Implementation
- [ ] Implement Pattern 3.3.1 (Load-Sync-Compute-Store)
  ```mojo
  @kernel
  fn matmul_gpu_tiled(
      C: DTypePointer[float32],
      A: DTypePointer[float32],
      B: DTypePointer[float32], 
      M: Int, N: Int, K: Int
  ):
      shared a_tile = Shared[float32, TILE_DIM, TILE_DIM]()
      shared b_tile = Shared[float32, TILE_DIM, TILE_DIM]()
      # Tiled implementation with barrier synchronization
  ```
- [ ] Test with different TILE_DIM values (16, 32, 64)
- [ ] Benchmark against naive implementation

#### Step 3.3: MLA Kernel GPU Adaptation
- [ ] Adapt attention mechanism for GPU execution
- [ ] Implement GPU-friendly softmax computation
- [ ] Optimize for 384-dim vs 768-dim embeddings

### Phase 4: Integration & Compatibility (3 hours)

#### Step 4.1: Hybrid CPU/GPU Support
- [ ] Create unified interface supporting both backends
  ```mojo
  struct HybridSearchEngine:
      var cpu_engine: SemanticSearchEngine
      var gpu_engine: GPUSemanticSearchEngine
      var use_gpu: Bool
      
      fn search(self, query: String) -> List[SearchResult]:
          if self.use_gpu and gpu_available():
              return self.gpu_engine.search(query)
          return self.cpu_engine.search(query)
  ```
- [ ] Implement automatic hardware detection
- [ ] Create performance-based routing logic

#### Step 4.2: Embedding Dimension Bridge
- [ ] Support both 384-dim and 768-dim models
- [ ] Create dimension adapter for compatibility
- [ ] Implement efficient dimension reduction if needed

#### Step 4.3: Onedev Integration Preservation
- [ ] Ensure GPU implementation maintains MCP tool compatibility
- [ ] Update OnedevBridge for GPU-accelerated operations
- [ ] Test portfolio intelligence features with GPU backend

### Phase 5: Autotuning Implementation (2 hours)

#### Step 5.1: Autotuning Framework
- [ ] Implement Pattern 4.5 (Autotuned Kernel)
  ```mojo
  @adaptive
  fn matmul_autotuned(...):
      alias TILE_DIM = autotune(8, 16, 32, 64)
      # Automated parameter selection
  ```
- [ ] Create evaluation function for tile sizes
- [ ] Implement search space exploration

#### Step 5.2: Hardware-Specific Optimization
- [ ] Profile on different GPU architectures
- [ ] Create hardware configuration profiles
- [ ] Document optimal parameters per GPU type

### Phase 6: Performance Validation (2 hours)

#### Step 6.1: Comprehensive Benchmarking
- [ ] Create benchmark suite comparing:
  - CPU (current): 12.7ms
  - GPU naive: Target 30-40ms
  - GPU tiled: Target 15-20ms
  - GPU autotuned: Target < 15ms
- [ ] Test with varying corpus sizes (10k, 50k, 100k+ snippets)
- [ ] Measure memory bandwidth utilization

#### Step 6.2: Scalability Testing
- [ ] Test multi-GPU scaling potential
- [ ] Benchmark concurrent query handling
- [ ] Analyze GPU memory limitations

### Phase 7: Production Integration (2 hours)

#### Step 7.1: Deployment Strategy
- [ ] Create GPU-enabled Docker images
- [ ] Update deployment scripts for GPU instances
- [ ] Implement health checks for GPU availability

#### Step 7.2: Monitoring & Observability
- [ ] Add GPU utilization metrics
- [ ] Implement kernel performance tracking
- [ ] Create alerts for GPU-specific issues

#### Step 7.3: Documentation Update
- [ ] Update README with GPU setup instructions
- [ ] Document performance characteristics
- [ ] Create migration guide for users

## Risk Mitigation Strategies

### Technical Risks
1. **GPU Memory Limitations**
   - Mitigation: Implement batch processing for large corpora
   - Fallback: Automatic CPU routing when GPU memory exhausted

2. **Dimension Compatibility**
   - Mitigation: Support multiple models simultaneously
   - Fallback: On-the-fly dimension adaptation

3. **Hardware Availability**
   - Mitigation: Maintain CPU implementation as fallback
   - Fallback: Seamless CPU/GPU switching

### Performance Risks
1. **GPU Overhead for Small Queries**
   - Mitigation: Intelligent routing based on query complexity
   - Fallback: Use CPU for simple queries

2. **Initial GPU Performance Regression**
   - Mitigation: Progressive optimization approach
   - Fallback: Maintain current CPU performance as baseline

## Success Criteria

### Must Have
- [ ] GPU kernels functional and correct
- [ ] Performance at least matches current CPU (12.7ms)
- [ ] Maintains all existing functionality
- [ ] Seamless CPU/GPU switching

### Should Have
- [ ] GPU performance < 10ms for 100k+ snippets
- [ ] Autotuning for optimal tile sizes
- [ ] Multi-GPU scaling capability

### Nice to Have
- [ ] Support for multiple embedding dimensions
- [ ] Dynamic kernel selection based on workload
- [ ] Cloud deployment automation

## Timeline Estimate

- **Phase 1-2**: 3 hours (Analysis & Setup)
- **Phase 3-4**: 7 hours (Core Implementation)
- **Phase 5-6**: 4 hours (Optimization & Validation)
- **Phase 7**: 2 hours (Production Ready)

**Total**: 16 hours of focused development

## Conclusion

This migration plan provides a systematic approach to evolving the current CPU-based implementation to leverage GPU acceleration while:
1. Preserving the excellent 12.7ms performance as baseline
2. Maintaining all onedev integration and portfolio intelligence
3. Adding GPU capabilities for even better scalability
4. Ensuring backward compatibility and graceful fallbacks

The key insight is that our current CPU implementation already exceeds plan-3's performance target, so the GPU migration focuses on scalability and handling larger corpora rather than just raw speed improvement.