"""
Working Performance Monitoring and Metrics Collection
Real-time monitoring for hybrid CPU/GPU semantic search system
"""

fn collect_search_metrics(
    query: String, 
    corpus_size: Int, 
    backend: String, 
    latency_ms: Float64,
    mcp_overhead_ms: Float64
):
    """Collect comprehensive performance metrics for each search operation."""
    print("📊 Performance Metrics Collection")
    print("================================")
    
    # Core performance metrics
    print("🔍 Search Operation Metrics:")
    print("  - Query:", query)
    print("  - Corpus size:", corpus_size, "snippets")
    print("  - Backend used:", backend)
    print("  - Core latency:", latency_ms, "ms")
    print("  - MCP overhead:", mcp_overhead_ms, "ms")
    print("  - Total latency:", latency_ms + mcp_overhead_ms, "ms")
    
    # Performance targets validation
    var target_latency = 20.0
    var total_latency = latency_ms + mcp_overhead_ms
    var performance_ratio = total_latency / target_latency
    
    print("\n🎯 Performance Target Analysis:")
    print("  - Target: <", target_latency, "ms")
    print("  - Achieved:", total_latency, "ms")
    print("  - Performance ratio:", performance_ratio)
    
    if performance_ratio <= 1.0:
        print("  ✅ Performance target met")
    else:
        print("  ⚠️  Performance target exceeded")
    
    # Backend efficiency metrics
    calculate_backend_efficiency(backend, corpus_size, latency_ms)

fn calculate_backend_efficiency(backend: String, corpus_size: Int, latency_ms: Float64):
    """Calculate backend-specific efficiency metrics."""
    print("\n⚡ Backend Efficiency Analysis:")
    print("  - Backend:", backend)
    
    # Calculate throughput
    var throughput = Float64(corpus_size) / (latency_ms / 1000.0)
    print("  - Throughput:", throughput, "snippets/second")
    
    # Backend-specific analysis
    if backend == "CPU_MLA_BMM":
        print("  - CPU optimization: Standard MLA+BMM kernels")
        print("  - Expected range: 5,000-50,000 snippets/second")
    elif backend == "GPU_Tiled_Pattern_3_3_1":
        print("  - GPU optimization: Shared memory tiling")
        print("  - Expected range: 50,000-500,000 snippets/second")
    elif backend == "GPU_Autotuned":
        print("  - GPU optimization: Autotuned for hardware")
        print("  - Expected range: 100,000-1,000,000 snippets/second")
    else:
        print("  - Backend: Unknown/Custom")

fn system_health_check(cpu_usage: Float64, memory_usage: Float64, gpu_utilization: Float64):
    """Perform system health monitoring."""
    print("\n🏥 System Health Check")
    print("======================")
    
    print("📊 Resource Utilization:")
    print("  - CPU usage:", cpu_usage, "%")
    print("  - Memory usage:", memory_usage, "%")
    print("  - GPU utilization:", gpu_utilization, "%")
    
    # Health status determination
    var health_status = "Healthy"
    
    if cpu_usage > 80.0 or memory_usage > 80.0 or gpu_utilization > 80.0:
        health_status = "Warning"
    
    if cpu_usage > 95.0 or memory_usage > 95.0 or gpu_utilization > 95.0:
        health_status = "Critical"
    
    print("\n🎯 System Health Status:", health_status)
    
    # Provide recommendations
    if health_status == "Warning":
        print("⚠️  Recommendations:")
        if cpu_usage > 80.0:
            print("  - Consider CPU optimization or scaling")
        if memory_usage > 80.0:
            print("  - Monitor memory leaks or increase RAM")
        if gpu_utilization > 80.0:
            print("  - Consider GPU load balancing")
    
    elif health_status == "Critical":
        print("🚨 Critical Issues:")
        if cpu_usage > 95.0:
            print("  - CPU bottleneck detected")
        if memory_usage > 95.0:
            print("  - Memory exhaustion risk")
        if gpu_utilization > 95.0:
            print("  - GPU overutilization")

fn onedev_mcp_performance_analysis(
    mcp_tool_calls: Int,
    mcp_latency_ms: Float64,
    mcp_success_rate: Float64
):
    """Analyze onedev MCP tool performance."""
    print("\n🔗 Onedev MCP Performance Analysis")
    print("===================================")
    
    print("📊 MCP Tool Metrics:")
    print("  - Tool calls made:", mcp_tool_calls)
    print("  - Average latency:", mcp_latency_ms, "ms")
    print("  - Success rate:", mcp_success_rate * 100.0, "%")
    
    # Performance evaluation
    var mcp_performance = "Good"
    
    if mcp_latency_ms > 100.0 or mcp_success_rate < 0.95:
        mcp_performance = "Needs Improvement"
    
    if mcp_latency_ms > 500.0 or mcp_success_rate < 0.90:
        mcp_performance = "Poor"
    
    print("🎯 MCP Performance Rating:", mcp_performance)
    
    # Provide optimization suggestions
    if mcp_performance != "Good":
        print("💡 Optimization Suggestions:")
        if mcp_latency_ms > 100.0:
            print("  - Optimize MCP tool response times")
            print("  - Consider caching frequently used data")
        if mcp_success_rate < 0.95:
            print("  - Investigate MCP tool failures")
            print("  - Implement better error handling")

fn generate_performance_report(
    total_searches: Int,
    avg_latency: Float64,
    hit_rate: Float64,
    backend_distribution: String
):
    """Generate comprehensive performance report."""
    print("\n📋 Performance Report Summary")
    print("=============================")
    
    print("📈 Aggregate Metrics:")
    print("  - Total searches:", total_searches)
    print("  - Average latency:", avg_latency, "ms")
    print("  - Cache hit rate:", hit_rate * 100.0, "%")
    print("  - Backend distribution:", backend_distribution)
    
    # Performance grade calculation
    var performance_grade = "A"
    
    if avg_latency > 20.0 or hit_rate < 0.80:
        performance_grade = "B"
    
    if avg_latency > 50.0 or hit_rate < 0.60:
        performance_grade = "C"
    
    if avg_latency > 100.0 or hit_rate < 0.40:
        performance_grade = "D"
    
    print("\n🏆 Overall Performance Grade:", performance_grade)
    
    # Recommendations based on grade
    if performance_grade != "A":
        print("\n📝 Improvement Recommendations:")
        if avg_latency > 20.0:
            print("  - Optimize search algorithms")
            print("  - Consider GPU acceleration")
        if hit_rate < 0.80:
            print("  - Improve caching strategies")
            print("  - Increase cache size if memory allows")

# Test performance monitoring
fn test_performance_monitoring():
    """Test the performance monitoring system."""
    print("🧪 Testing Performance Monitoring System")
    print("========================================")
    
    # Test search metrics collection
    collect_search_metrics(
        "authentication patterns",
        15000,
        "CPU_MLA_BMM",
        12.7,
        4.3
    )
    
    # Test system health check
    system_health_check(65.2, 78.5, 23.1)
    
    # Test MCP performance analysis
    onedev_mcp_performance_analysis(45, 67.3, 0.96)
    
    # Test performance report generation
    generate_performance_report(
        1250,
        18.4,
        0.82,
        "60% CPU, 40% GPU"
    )
    
    print("\n✅ Performance monitoring system test completed!")

fn main():
    """Test the performance monitoring system."""
    test_performance_monitoring()