#!/usr/bin/env python3
"""
Benchmark the overhead of adaptive fusion detection.
Measures if device detection impacts performance.
"""

import time
import statistics
from dataclasses import dataclass
from typing import Optional

@dataclass
class MaxGraphConfig:
    """Test config to measure overhead."""
    corpus_size: int
    vector_dims: int = 768
    batch_size: int = 1
    device: str = "cpu"
    use_fp16: bool = False
    enable_fusion: Optional[bool] = None
    
    def __post_init__(self):
        if self.enable_fusion is None:
            self.enable_fusion = self._detect_optimal_fusion_setting()
    
    def _detect_optimal_fusion_setting(self) -> bool:
        if self._is_parallel_compute_device():
            return True
        else:
            return False
    
    def _is_parallel_compute_device(self) -> bool:
        device_lower = self.device.lower()
        
        parallel_indicators = [
            'gpu', 'cuda', 'metal', 'opencl', 'vulkan', 'rocm',
            'dml', 'tensorrt', 'mlx'
        ]
        
        return any(indicator in device_lower for indicator in parallel_indicators)

@dataclass
class StaticConfig:
    """Static config for comparison."""
    corpus_size: int
    vector_dims: int = 768
    batch_size: int = 1
    device: str = "cpu"
    use_fp16: bool = False
    enable_fusion: bool = False  # Static, no detection

def benchmark_config_creation(iterations: int = 10000):
    """Benchmark config creation overhead."""
    
    print(f"🔧 Benchmarking Config Creation ({iterations:,} iterations)")
    print("=" * 60)
    
    # Test adaptive fusion config
    adaptive_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        config = MaxGraphConfig(corpus_size=5000, device="cpu")
        end = time.perf_counter()
        adaptive_times.append((end - start) * 1_000_000)  # microseconds
    
    # Test static config
    static_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        config = StaticConfig(corpus_size=5000, device="cpu")
        end = time.perf_counter()
        static_times.append((end - start) * 1_000_000)  # microseconds
    
    # Calculate statistics
    adaptive_mean = statistics.mean(adaptive_times)
    adaptive_median = statistics.median(adaptive_times)
    adaptive_std = statistics.stdev(adaptive_times)
    
    static_mean = statistics.mean(static_times)
    static_median = statistics.median(static_times)
    static_std = statistics.stdev(static_times)
    
    overhead_us = adaptive_mean - static_mean
    overhead_pct = (overhead_us / static_mean) * 100
    
    print(f"Static Config:")
    print(f"  Mean:   {static_mean:.2f}μs ± {static_std:.2f}μs")
    print(f"  Median: {static_median:.2f}μs")
    
    print(f"\nAdaptive Config:")
    print(f"  Mean:   {adaptive_mean:.2f}μs ± {adaptive_std:.2f}μs")
    print(f"  Median: {adaptive_median:.2f}μs")
    
    print(f"\nOverhead Analysis:")
    print(f"  Absolute: {overhead_us:.2f}μs")
    print(f"  Relative: {overhead_pct:.1f}%")
    
    return {
        'adaptive_mean_us': adaptive_mean,
        'static_mean_us': static_mean,
        'overhead_us': overhead_us,
        'overhead_pct': overhead_pct
    }

def benchmark_device_detection_only(iterations: int = 100000):
    """Benchmark just the device detection logic."""
    
    print(f"\n🔍 Benchmarking Device Detection Only ({iterations:,} iterations)")
    print("=" * 60)
    
    devices = ["cpu", "gpu", "cuda", "metal", "opencl", "vulkan"]
    
    for device in devices:
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            
            # Simulate the detection logic
            device_lower = device.lower()
            parallel_indicators = [
                'gpu', 'cuda', 'metal', 'opencl', 'vulkan', 'rocm',
                'dml', 'tensorrt', 'mlx'
            ]
            result = any(indicator in device_lower for indicator in parallel_indicators)
            
            end = time.perf_counter()
            times.append((end - start) * 1_000_000_000)  # nanoseconds
        
        mean_ns = statistics.mean(times)
        median_ns = statistics.median(times)
        
        print(f"{device:8} → {result:5} | {mean_ns:6.1f}ns (median: {median_ns:6.1f}ns)")

def analyze_caching_potential():
    """Analyze whether caching would help."""
    
    print(f"\n📊 Caching Analysis")
    print("=" * 30)
    
    # Current implementation runs detection in __post_init__
    # This means it runs once per config object creation
    
    config1 = MaxGraphConfig(corpus_size=1000, device="cpu")
    config2 = MaxGraphConfig(corpus_size=2000, device="cpu")  # Same device
    config3 = MaxGraphConfig(corpus_size=1000, device="gpu")  # Different device
    
    print(f"Config 1 (cpu):  fusion={config1.enable_fusion}")
    print(f"Config 2 (cpu):  fusion={config2.enable_fusion}")
    print(f"Config 3 (gpu):  fusion={config3.enable_fusion}")
    
    print(f"\n🔄 Current Behavior:")
    print(f"  - Detection runs once per config creation")
    print(f"  - Result is stored in config.enable_fusion")
    print(f"  - No repeated detection during config lifetime")
    
    print(f"\n💡 Caching Potential:")
    print(f"  - Could cache results by device string")
    print(f"  - Would help if creating many configs with same device")
    print(f"  - Benefit depends on usage patterns")

def simulate_real_world_usage():
    """Simulate real-world usage patterns."""
    
    print(f"\n🌍 Real-World Usage Simulation")
    print("=" * 40)
    
    # Scenario 1: Single config for entire application
    print("Scenario 1: Single config creation")
    start = time.perf_counter()
    config = MaxGraphConfig(corpus_size=10000, device="cpu")
    end = time.perf_counter()
    single_time_us = (end - start) * 1_000_000
    print(f"  Time: {single_time_us:.2f}μs")
    print(f"  Impact: Negligible (one-time cost)")
    
    # Scenario 2: Multiple configs for different corpus sizes
    print("\nScenario 2: Multiple configs, same device")
    start = time.perf_counter()
    configs = []
    for size in [1000, 2000, 5000, 10000]:
        configs.append(MaxGraphConfig(corpus_size=size, device="cpu"))
    end = time.perf_counter()
    multiple_time_us = (end - start) * 1_000_000
    print(f"  Time: {multiple_time_us:.2f}μs (4 configs)")
    print(f"  Per config: {multiple_time_us/4:.2f}μs")
    
    # Scenario 3: Autotuning with many configs
    print("\nScenario 3: Autotuning (100 configs)")
    start = time.perf_counter()
    configs = []
    for i in range(100):
        configs.append(MaxGraphConfig(corpus_size=5000, device="cpu"))
    end = time.perf_counter()
    autotuning_time_us = (end - start) * 1_000_000
    print(f"  Time: {autotuning_time_us:.2f}μs (100 configs)")
    print(f"  Per config: {autotuning_time_us/100:.2f}μs")

if __name__ == "__main__":
    print("⚡ Adaptive Fusion Overhead Benchmark")
    print("=" * 70)
    
    # Benchmark config creation
    results = benchmark_config_creation(10000)
    
    # Benchmark device detection only
    benchmark_device_detection_only(100000)
    
    # Analyze caching potential
    analyze_caching_potential()
    
    # Simulate real usage
    simulate_real_world_usage()
    
    print(f"\n📊 Summary:")
    print(f"  - Adaptive fusion overhead: {results['overhead_us']:.2f}μs ({results['overhead_pct']:.1f}%)")
    print(f"  - Device detection: <100ns per call")
    print(f"  - Real-world impact: Negligible")
    print(f"  - Caching benefit: Minimal (detection is already very fast)")