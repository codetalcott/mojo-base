# Autotuning V2: Complete Implementation Plan

## 🎯 Executive Summary

**Objective**: Run accurate GPU autotuning on Lambda Cloud using our **fixed, working Mojo kernels** with **production-scale data** to get **real performance metrics**.

**Why V2 is Critical**: Previous autotuning was entirely simulated and used broken code. All previous benchmarks are invalid.

## 📋 Implementation Phases

### **Phase 1: Real Benchmark Infrastructure**

#### 1.1 Enhanced Integration Test Framework
Extend our working `integration_test_complete.mojo` with real GPU benchmarking:

```mojo
// Add to integration_test_complete.mojo
fn benchmark_real_kernel_performance(
    corpus_size: Int,
    vector_dims: Int,
    tile_size: Int, 
    block_size: Int,
    shared_memory_kb: Int
) -> KernelBenchmarkResult:
    """Benchmark actual GPU kernel performance with real data."""
    
    // Create real test data
    var test_corpus = generate_test_vectors(corpus_size, vector_dims)
    var test_queries = generate_test_queries(100, vector_dims)
    
    // Initialize our ACTUAL working kernels
    var bmm_kernel = OptimizedBMMKernel(corpus_size)
    var mla_kernel = OptimizedMLAKernel(corpus_size) 
    
    // Configure kernel parameters
    bmm_kernel.set_tile_size(tile_size)
    bmm_kernel.set_block_size(block_size)
    bmm_kernel.set_shared_memory(shared_memory_kb * 1024)
    
    // Measure real performance
    var start_time = time.now()
    var results = bmm_kernel.batch_similarity_search(test_queries)
    var end_time = time.now()
    
    return KernelBenchmarkResult(
        latency_ms=Float64(end_time - start_time) / 1_000_000,
        throughput=calculate_throughput(corpus_size, queries.size(), latency_ms),
        memory_bandwidth=measure_memory_usage(),
        gpu_occupancy=get_gpu_metrics()
    )
```

#### 1.2 Production-Scale Test Data
```mojo
fn generate_production_scale_corpus() -> TestCorpus:
    """Generate realistic test corpus matching production requirements."""
    return TestCorpus(
        vector_count=50_000,      // Production scale
        vector_dimensions=768,    // Real embedding size  
        categories=["auth", "api", "db", "ui", "algorithms"],
        languages=["python", "typescript", "mojo", "javascript"]
    )
```

### **Phase 2: Lambda Cloud Deployment**

#### 2.1 Deploy Fixed Kernels
```bash
# Deploy our working, fixed kernels to Lambda Cloud
rsync -av src/kernels/ ubuntu@lambda-gpu:/home/ubuntu/mojo-kernels/
rsync -av integration_test_complete.mojo ubuntu@lambda-gpu:/home/ubuntu/
```

#### 2.2 Lambda Cloud Test Script
```python
# lambda_autotuning_v2.py
import subprocess
import json
import time
from typing import Dict, List, Any

class LambdaAutotuningV2:
    """Real GPU autotuning on Lambda Cloud with fixed kernels."""
    
    def __init__(self):
        self.gpu_instance = "lambda-gpu-a10"
        self.test_configurations = self.generate_test_matrix()
        
    def generate_test_matrix(self) -> List[Dict]:
        """Generate comprehensive test matrix for real GPU testing."""
        configurations = []
        
        # Production-focused parameter ranges
        tile_sizes = [16, 32, 48, 64, 96, 128]
        block_sizes = [32, 64, 128, 256]
        memory_configs = [4, 8, 16, 32]  # KB
        
        for tile in tile_sizes:
            for block in block_sizes:
                for memory in memory_configs:
                    configurations.append({
                        'tile_size': tile,
                        'block_size': block, 
                        'shared_memory_kb': memory,
                        'corpus_size': 50_000,
                        'vector_dims': 768
                    })
                    
        return configurations
    
    def run_single_benchmark(self, config: Dict) -> Dict:
        """Run single benchmark configuration on real GPU."""
        
        # Execute real Mojo kernel on Lambda GPU
        cmd = [
            'pixi', 'run', 'mojo', 'integration_test_complete.mojo',
            '--benchmark-mode',
            f'--tile-size={config["tile_size"]}',
            f'--block-size={config["block_size"]}', 
            f'--memory-kb={config["shared_memory_kb"]}',
            f'--corpus-size={config["corpus_size"]}',
            f'--vector-dims={config["vector_dims"]}'
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            return {
                'config': config,
                'success': False,
                'error': result.stderr,
                'execution_time': end_time - start_time
            }
            
        # Parse real performance metrics
        performance_data = self.parse_benchmark_output(result.stdout)
        
        return {
            'config': config,
            'success': True,
            'performance': performance_data,
            'execution_time': end_time - start_time
        }
    
    def run_comprehensive_autotuning(self) -> Dict:
        """Run comprehensive autotuning with all configurations."""
        print("🚀 Starting Autotuning V2 - Real GPU Performance Testing")
        print(f"📊 Testing {len(self.test_configurations)} configurations")
        print(f"🔧 Using fixed Mojo kernels with production-scale data")
        
        results = []
        best_config = None
        best_performance = float('inf')
        
        for i, config in enumerate(self.test_configurations):
            print(f"\n[{i+1:3d}/{len(self.test_configurations)}] Testing: {config}")
            
            # Run real benchmark
            result = self.run_single_benchmark(config)
            results.append(result)
            
            if result['success']:
                latency = result['performance']['latency_ms']
                if latency < best_performance:
                    best_performance = latency
                    best_config = result
                    
                print(f"   ✅ Latency: {latency:.2f}ms | "
                      f"Throughput: {result['performance']['throughput']:.1f} vec/sec")
            else:
                print(f"   ❌ Failed: {result['error'][:100]}...")
        
        return {
            'total_tests': len(results),
            'successful_tests': sum(1 for r in results if r['success']),
            'best_config': best_config,
            'all_results': results,
            'summary': {
                'best_latency_ms': best_performance,
                'success_rate': sum(1 for r in results if r['success']) / len(results),
                'test_duration_hours': sum(r['execution_time'] for r in results) / 3600
            }
        }
```

### **Phase 3: Performance Analysis & Validation**

#### 3.1 Real vs Simulated Comparison
```python
def compare_v1_vs_v2_results():
    """Compare previous simulated results with real GPU measurements."""
    
    v1_results = load_json('autotuning_results/autotune_20250702_233614_results.json')
    v2_results = load_json('autotuning_results/autotune_v2_real_gpu_results.json')
    
    print("🔍 V1 vs V2 Autotuning Comparison")
    print("=" * 50)
    print(f"V1 (Simulated):  {v1_results['best_latency']:.2f}ms")
    print(f"V2 (Real GPU):   {v2_results['best_latency']:.2f}ms") 
    print(f"Reality Check:   {v2_results['best_latency'] / v1_results['best_latency']:.1f}x slower")
    
    # Detailed analysis
    print("\n📊 Detailed Analysis:")
    print(f"V1 Corpus: {v1_results['corpus_size']:,} vectors ({v1_results['vector_dims']}D)")
    print(f"V2 Corpus: {v2_results['corpus_size']:,} vectors ({v2_results['vector_dims']}D)")
    print(f"Scale Factor: {v2_results['corpus_size'] / v1_results['corpus_size']:.1f}x larger")
```

## 🛠️ Technical Implementation Details

### **Enhanced Integration Test**
Modify `integration_test_complete.mojo` to support benchmark mode:

```mojo
fn main():
    """Enhanced integration test with benchmark mode support."""
    
    // Check for benchmark mode arguments
    var benchmark_mode = check_benchmark_args()
    
    if benchmark_mode:
        // Run performance benchmarking
        run_kernel_benchmarks()
    else:
        // Run standard integration tests
        test_end_to_end_pipeline()
        test_error_handling()
        test_scalability_simulation()
    
    print("🎯 Integration Test Suite Complete!")
```

### **Real Performance Metrics**
```mojo
struct RealPerformanceMetrics:
    """Real GPU performance measurements."""
    var latency_ms: Float64
    var throughput_vectors_per_sec: Float64
    var memory_bandwidth_gb_per_sec: Float64
    var gpu_occupancy_percent: Float64
    var power_consumption_watts: Float64
    var thermal_throttling: Bool
    
    fn __init__(out self):
        self.latency_ms = 0.0
        self.throughput_vectors_per_sec = 0.0
        self.memory_bandwidth_gb_per_sec = 0.0
        self.gpu_occupancy_percent = 0.0
        self.power_consumption_watts = 0.0
        self.thermal_throttling = False
```

## 📈 Expected Results

### **Realistic Performance Expectations**
Based on our fixed kernels and production scale:

- **Latency Range**: 15-75ms (much more realistic than 2.99ms simulation)
- **Throughput**: 1,000-10,000 vectors/second
- **Memory Efficiency**: 60-85% bandwidth utilization
- **GPU Occupancy**: 70-95%

### **Validation Criteria**
✅ **Real < 50ms latency** for 50K vector corpus  
✅ **Sustained throughput** > 2,000 vectors/second  
✅ **Memory efficiency** > 70%  
✅ **No thermal throttling** under normal load  
✅ **End-to-end pipeline** < 100ms total latency  

## 🚀 Execution Timeline

### **Week 1: Infrastructure**
- ✅ Enhanced integration test framework
- ✅ Production-scale test data generation  
- ✅ Lambda Cloud deployment scripts

### **Week 2: Testing**
- 🔄 Comprehensive parameter sweep
- 🔄 Real GPU performance measurement
- 🔄 Results analysis and optimization

### **Week 3: Validation**  
- 🔄 End-to-end pipeline testing
- 🔄 Production readiness validation
- 🔄 Documentation and deployment

## 🎯 Success Metrics

**V2 Success = First Accurate Performance Data for Production System**

1. **Real Hardware**: Actual A10 GPU execution ✅
2. **Production Scale**: 50K+ vectors, 768D ✅  
3. **Fixed Kernels**: Using working, modern Mojo code ✅
4. **End-to-End**: Full pipeline performance ✅
5. **Comprehensive**: Complete parameter space ✅

**Bottom Line**: V2 will finally give us trustworthy performance data to make production deployment decisions.